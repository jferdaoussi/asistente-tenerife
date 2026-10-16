"""Herramienta ``get_weather``: pronóstico del tiempo en un lugar de la guía.

Usa la API de pronóstico de Open-Meteo (gratuita y sin clave). El reparto de
trabajo es el siguiente:

- El LLM interpreta el lenguaje ("el sábado en Masca") y pide la herramienta
  con argumentos estructurados (``fecha="2026-10-17"``, ``lugar="Masca"``).
- Este módulo no se fía de esos argumentos: los valida con Pydantic (formato,
  rango de fechas con pronóstico, lugar conocido), llama a la API con
  reintentos y devuelve solo los datos útiles para un turista.

``get_weather`` nunca lanza excepciones: cualquier fallo se devuelve como un
diccionario ``{"ok": False, "error": ..., "mensaje": ...}`` para que el LLM
pueda explicarlo, corregir sus argumentos o preguntar al usuario. Cada intento
queda en el registro de herramientas (``logs/tools.jsonl``).
"""

import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from src.config import WEATHER, WeatherConfig
from src.tool_log import log_event

TOOL_NAME = "get_weather"
TOOL_DESCRIPTION = (
    "Pronóstico del tiempo para un día y un lugar concretos de Tenerife: "
    "estado del cielo, temperaturas, sensación térmica, lluvia, viento, índice "
    "UV, humedad y horas de amanecer y atardecer. Úsala cuando el usuario "
    "pregunte por el tiempo o cuando sea útil para recomendar un plan."
)

#: Lugares de la guía con sus coordenadas (latitud, longitud). Tenerife tiene
#: microclimas muy marcados, así que el tiempo se consulta en cada sitio y no
#: en un punto genérico de la isla. Coordenadas obtenidas con la API de
#: geocodificación de Open-Meteo o, para parajes sin municipio (miradores,
#: Teide, Punta de Teno), del punto de visita que menciona la guía.
PLACES: dict[str, tuple[float, float]] = {
    # Zona Norte
    "Santa Cruz de Tenerife": (28.4682, -16.2546),
    "San Cristóbal de La Laguna": (28.4853, -16.3201),
    "Anaga (Cruz del Carmen)": (28.5310, -16.2800),
    "Playa de Benijo": (28.5735, -16.1876),
    "La Orotava": (28.3908, -16.5231),
    "Puerto de la Cruz": (28.4169, -16.5509),
    "Icod de los Vinos": (28.3724, -16.7119),
    "Garachico": (28.3737, -16.7634),
    "Punta de Teno": (28.3425, -16.9224),
    "Teide (Las Cañadas)": (28.2232, -16.6267),
    "Teide (cumbre)": (28.2724, -16.6425),
    # Zona Sur
    "Costa Adeje": (28.0870, -16.7330),
    "Los Cristianos": (28.0501, -16.7201),
    "Los Gigantes": (28.2436, -16.8415),
    "Masca": (28.3040, -16.8424),
    "El Médano": (28.0462, -16.5363),
}

#: Valores admitidos para ``lugar``; en el esquema JSON aparecen como ``enum``.
Lugar = Literal[tuple(PLACES)]

#: Códigos meteorológicos WMO que devuelve Open-Meteo, traducidos a texto para
#: no depender de que el LLM los conozca.
WMO_CODES = {
    0: "Despejado",
    1: "Mayormente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna débil",
    53: "Llovizna moderada",
    55: "Llovizna intensa",
    56: "Llovizna helada",
    57: "Llovizna helada intensa",
    61: "Lluvia débil",
    63: "Lluvia moderada",
    65: "Lluvia fuerte",
    66: "Lluvia helada",
    67: "Lluvia helada intensa",
    71: "Nevada débil",
    73: "Nevada moderada",
    75: "Nevada fuerte",
    77: "Granos de nieve",
    80: "Chubascos débiles",
    81: "Chubascos moderados",
    82: "Chubascos violentos",
    85: "Chubascos de nieve",
    86: "Chubascos de nieve fuertes",
    95: "Tormenta",
    96: "Tormenta con granizo",
    99: "Tormenta con granizo fuerte",
}

#: Variables diarias que se piden a Open-Meteo y nombre con el que se
#: devuelven al LLM (con la unidad en el nombre para evitar ambigüedades).
DAILY_FIELDS = {
    "temperature_2m_max": "temperatura_max_c",
    "temperature_2m_min": "temperatura_min_c",
    "apparent_temperature_max": "sensacion_max_c",
    "apparent_temperature_min": "sensacion_min_c",
    "precipitation_probability_max": "prob_lluvia_pct",
    "precipitation_sum": "lluvia_mm",
    "wind_speed_10m_max": "viento_max_kmh",
    "wind_gusts_10m_max": "rachas_max_kmh",
    "uv_index_max": "uv_max",
    "relative_humidity_2m_mean": "humedad_media_pct",
}

#: Códigos HTTP que indican un fallo pasajero del servicio: se reintenta.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def today(config: WeatherConfig = WEATHER) -> date:
    """Fecha de hoy en Canarias (no en la zona horaria del ordenador)."""
    return datetime.now(ZoneInfo(config.timezone)).date()


def _normalize(text: str) -> str:
    """Pasa un nombre a minúsculas y sin tildes para compararlo."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    return " ".join(ascii_text.decode().lower().split())


_PLACE_INDEX = {_normalize(name): name for name in PLACES}


class WeatherRequest(BaseModel):
    """Argumentos de ``get_weather``.

    Su esquema JSON es el que se le da al LLM (ver ``src/tools.py``), de modo
    que el esquema y la validación salen de la misma definición.
    """

    fecha: date = Field(
        description=(
            "Día del pronóstico en formato AAAA-MM-DD. Solo hay pronóstico "
            f"desde hoy hasta dentro de {WEATHER.max_days_ahead} días."
        )
    )
    lugar: Lugar = Field(
        description=(
            "Sitio de Tenerife indicado por el usuario o el más cercano a él. "
            "El tiempo cambia mucho entre zonas de la isla: si el usuario no "
            "ha indicado sitio ni zona, no elijas uno; pregúntale primero."
        )
    )

    @field_validator("lugar", mode="before")
    @classmethod
    def _canonical_place(cls, value: Any) -> Any:
        """Acepta variantes sin tildes o en minúsculas ("masca", "el medano")."""
        if isinstance(value, str):
            return _PLACE_INDEX.get(_normalize(value), value)
        return value

    @field_validator("fecha")
    @classmethod
    def _within_forecast_range(cls, value: date) -> date:
        """Comprueba que la fecha tenga pronóstico (de hoy a hoy + N días)."""
        hoy = today()
        last = hoy + timedelta(days=WEATHER.max_days_ahead)
        if value < hoy:
            raise PydanticCustomError(
                "fecha_pasada",
                "El {fecha} ya ha pasado: solo hay pronósticos desde hoy ({hoy}).",
                {"fecha": value.isoformat(), "hoy": hoy.isoformat()},
            )
        if value > last:
            raise PydanticCustomError(
                "fecha_lejana",
                "Aún no hay pronóstico para el {fecha}: llega hasta el {ultimo}. "
                "Estará disponible a partir del {disponible}.",
                {
                    "fecha": value.isoformat(),
                    "ultimo": last.isoformat(),
                    "disponible": (
                        value - timedelta(days=WEATHER.max_days_ahead)
                    ).isoformat(),
                },
            )
        return value


class WeatherAPIError(Exception):
    """Fallo al consultar Open-Meteo.

    Attributes:
        transient: True si el fallo es pasajero y merece un reintento.
    """

    def __init__(self, message: str, transient: bool) -> None:
        """Guarda el mensaje y si el fallo es transitorio."""
        super().__init__(message)
        self.transient = transient


def _error(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    """Resultado de error con el formato común que recibe el LLM."""
    return {"ok": False, "error": kind, "mensaje": message, **extra}


def _validation_error(err: ValidationError) -> dict[str, Any]:
    """Traduce los errores de Pydantic a un resultado comprensible.

    Args:
        err: Error de validación de :class:`WeatherRequest`.

    Returns:
        Resultado de error con el tipo del primer fallo y los mensajes de
        todos los campos incorrectos.
    """
    kinds, messages, extra = [], [], {}
    for issue in err.errors():
        field = issue["loc"][0] if issue["loc"] else ""
        if issue["type"] in ("fecha_pasada", "fecha_lejana"):
            kinds.append(issue["type"])
            messages.append(issue["msg"])
        elif field == "fecha":
            kinds.append("formato_fecha")
            messages.append(
                f"Fecha no válida ({issue.get('input')!r}). Usa el formato "
                f"AAAA-MM-DD, p. ej. {today().isoformat()}."
            )
        elif field == "lugar":
            kinds.append("lugar_desconocido")
            messages.append(f"No tengo el lugar {issue.get('input')!r}.")
            extra["lugares_validos"] = list(PLACES)
        else:
            kinds.append("argumentos_invalidos")
            messages.append(f"{field or 'argumentos'}: {issue['msg']}")
    return _error(kinds[0], " ".join(messages), **extra)


def _request_forecast(
    lat: float, lon: float, day: date, config: WeatherConfig
) -> dict[str, Any]:
    """Hace una única petición a Open-Meteo.

    Raises:
        WeatherAPIError: Si la petición falla; indica si conviene reintentar.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(["weather_code", *DAILY_FIELDS, "sunrise", "sunset"]),
        "timezone": config.timezone,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    try:
        response = requests.get(config.api_url, params=params, timeout=config.timeout_s)
    except requests.Timeout as err:
        raise WeatherAPIError("Open-Meteo no respondió a tiempo", True) from err
    except requests.ConnectionError as err:
        raise WeatherAPIError("No se pudo conectar con Open-Meteo", True) from err
    if response.status_code in TRANSIENT_STATUS:
        raise WeatherAPIError(f"Open-Meteo devolvió HTTP {response.status_code}", True)
    if not response.ok:
        raise WeatherAPIError(
            f"Open-Meteo rechazó la petición (HTTP {response.status_code})", False
        )
    try:
        return response.json()
    except ValueError as err:
        raise WeatherAPIError(
            "Open-Meteo devolvió una respuesta ilegible", False
        ) from err


def _summarize(
    data: dict[str, Any], request: WeatherRequest, config: WeatherConfig
) -> dict[str, Any]:
    """Extrae del JSON de Open-Meteo los datos útiles para un turista.

    Raises:
        WeatherAPIError: Si la respuesta no tiene la forma esperada.
    """
    try:
        daily = data["daily"]
        code = daily["weather_code"][0]
        result = {
            "ok": True,
            "lugar": request.lugar,
            "fecha": request.fecha.isoformat(),
            "altitud_m": round(data["elevation"]),
            "cielo": WMO_CODES.get(code, f"Código meteorológico {code}"),
        }
        for api_name, name in DAILY_FIELDS.items():
            result[name] = daily[api_name][0]
        result["amanecer"] = daily["sunrise"][0][-5:]
        result["atardecer"] = daily["sunset"][0][-5:]
    except (KeyError, IndexError, TypeError) as err:
        raise WeatherAPIError("Respuesta de Open-Meteo incompleta", False) from err

    days_ahead = (request.fecha - today(config)).days
    if days_ahead > config.reliable_days:
        result["aviso"] = (
            f"Pronóstico a {days_ahead} días: orientativo, puede cambiar bastante."
        )
    result["fuente"] = "Open-Meteo"
    return result


def get_weather(
    fecha: str, lugar: str, config: WeatherConfig = WEATHER
) -> dict[str, Any]:
    """Pronóstico del tiempo para un día y un lugar de Tenerife.

    Args:
        fecha: Día en formato AAAA-MM-DD, desde hoy hasta hoy + 15 días.
        lugar: Uno de los lugares de :data:`PLACES` (se aceptan variantes sin
            tildes o en minúsculas).
        config: Parámetros de la API (URL, plazos, reintentos).

    Returns:
        Si todo va bien, ``{"ok": True, ...}`` con el estado del cielo,
        temperaturas, lluvia, viento, UV, humedad y horas de sol. Si algo
        falla, ``{"ok": False, "error": tipo, "mensaje": explicación}``. Tipos
        de error: ``formato_fecha``, ``fecha_pasada``, ``fecha_lejana``,
        ``lugar_desconocido``, ``argumentos_invalidos``,
        ``servicio_no_disponible`` y ``respuesta_invalida``.
    """
    args = {"fecha": fecha, "lugar": lugar}
    try:
        request = WeatherRequest(fecha=fecha, lugar=lugar)
    except ValidationError as err:
        result = _validation_error(err)
        log_event(
            TOOL_NAME,
            args=args,
            stage="validacion",
            ok=False,
            error=result["error"],
            message=result["mensaje"],
        )
        return result

    lat, lon = PLACES[request.lugar]
    for attempt in range(1, config.max_attempts + 1):
        start = time.perf_counter()
        try:
            result = _summarize(
                _request_forecast(lat, lon, request.fecha, config), request, config
            )
        except WeatherAPIError as err:
            log_event(
                TOOL_NAME,
                args=args,
                stage="api",
                attempt=attempt,
                ok=False,
                error=str(err),
                transient=err.transient,
                latency_ms=round((time.perf_counter() - start) * 1000),
            )
            if err.transient and attempt < config.max_attempts:
                time.sleep(config.retry_delay_s)
                continue
            kind = "servicio_no_disponible" if err.transient else "respuesta_invalida"
            return _error(
                kind,
                f"{err}. No puedo consultar el tiempo ahora mismo; "
                "inténtalo más tarde o consulta AEMET.",
            )
        log_event(
            TOOL_NAME,
            args=args,
            stage="api",
            attempt=attempt,
            ok=True,
            result=result,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
        return result
    raise AssertionError("max_attempts debe ser al menos 1")  # pragma: no cover
