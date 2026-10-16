"""Registro de las herramientas que el LLM puede invocar.

Cada herramienta se define con:

- un **nombre** y una **descripción**, que el LLM lee para decidir cuándo
  usarla;
- un **modelo Pydantic** con sus argumentos, del que sale el JSON Schema que
  recibe el LLM y con el que se validan los argumentos que este devuelve;
- una **función** que la ejecuta y devuelve un diccionario.

:class:`ToolRegistry` reúne las tres herramientas del asistente, genera sus
declaraciones para Gemini y las ejecuta de forma segura: un fallo nunca
detiene la conversación, sino que se devuelve al LLM como un resultado con
``"ok": False``. Cada llamada queda en ``logs/tools.jsonl``.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote_plus

from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from src import weather
from src.llm import LLMError
from src.rag import GuideIndex, build_index
from src.tool_log import log_event

logger = logging.getLogger(__name__)

SEARCH_DESCRIPTION = (
    "Busca en la guía de Tenerife (TENERIFE.pdf) los fragmentos más relevantes "
    "para una consulta: sitios que visitar, playas, miradores, rutas, "
    "restaurantes, ocio y consejos. Úsala siempre antes de recomendar o "
    "describir lugares, planes o comida. Devuelve el texto de cada fragmento "
    "con su fuente (página) para citarla."
)
ROUTE_DESCRIPTION = (
    "Genera un enlace de Google Maps con la ruta hasta un sitio de Tenerife "
    "desde la ubicación actual del usuario. Úsala cuando pregunte cómo llegar "
    "a un sitio; la guía no incluye enlaces propios."
)
MAPS_DIRECTIONS_URL = "https://www.google.com/maps/dir/?api=1&destination="


class SearchArgs(BaseModel):
    """Argumentos de ``buscar_guia``."""

    consulta: str = Field(
        min_length=3,
        max_length=300,
        description=(
            "Qué buscar, redactado como una pregunta completa y autónoma: con "
            "el nombre del sitio en lugar de referencias como 'allí' o 'ese "
            "pueblo'. Ejemplo: 'dónde comer cerca de Masca'."
        ),
    )
    zona: Literal["norte", "sur", "toda"] = Field(
        default="toda",
        description="Zona de la isla a la que limitar la búsqueda; 'toda' si "
        "el usuario no indica ninguna.",
    )


class RouteArgs(BaseModel):
    """Argumentos de ``como_llegar``."""

    sitio: str = Field(
        min_length=2,
        max_length=150,
        description="Sitio al que quiere ir el usuario, con el nombre que usa "
        "la guía. Ejemplo: 'Playa de Benijo'.",
    )


def llm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema de un modelo Pydantic, limpio para enviarlo al LLM.

    Quita los títulos que Pydantic genera automáticamente y el docstring de la
    clase, que están pensados para desarrolladores y solo gastarían tokens.

    Args:
        model: Modelo con los argumentos de una herramienta.

    Returns:
        Esquema con ``type``, ``properties`` y ``required``.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def _invalid_args(err: ValidationError) -> dict[str, Any]:
    """Resultado de error para argumentos que no cumplen el esquema."""
    details = "; ".join(
        f"{'.'.join(map(str, issue['loc'])) or 'argumentos'}: {issue['msg']}"
        for issue in err.errors()
    )
    return {"ok": False, "error": "argumentos_invalidos", "mensaje": details}


def _log_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Versión compacta de un resultado para el registro.

    Los fragmentos de la guía se sustituyen por sus fuentes, para que el
    registro no repita miles de caracteres de texto en cada búsqueda.
    """
    if "fragmentos" not in result:
        return result
    summary = dict(result)
    summary["fragmentos"] = [fragment["fuente"] for fragment in result["fragmentos"]]
    return summary


@dataclass(frozen=True)
class Tool:
    """Herramienta disponible para el LLM.

    Attributes:
        name: Nombre con el que la invoca el LLM.
        description: Cuándo y para qué usarla (la lee el LLM).
        args_model: Modelo Pydantic de sus argumentos.
        run: Función que recibe los argumentos y devuelve el resultado.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[dict[str, Any]], dict[str, Any]]

    def declaration(self) -> dict[str, Any]:
        """Declaración de la herramienta en JSON Schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": llm_schema(self.args_model),
        }


class ToolRegistry:
    """Herramientas del asistente: declaración y ejecución segura.

    Attributes:
        tools: Herramientas disponibles, por nombre.
    """

    def __init__(self, index: GuideIndex | None = None):
        """Crea el registro.

        Args:
            index: Índice de la guía para ``buscar_guia``. Si se omite, se
                construye (o se reutiliza el de disco) la primera vez que se
                necesita.
        """
        self._index = index
        self.tools = {
            tool.name: tool
            for tool in (
                Tool("buscar_guia", SEARCH_DESCRIPTION, SearchArgs, self._search),
                Tool(
                    weather.TOOL_NAME,
                    weather.TOOL_DESCRIPTION,
                    weather.WeatherRequest,
                    self._weather,
                ),
                Tool("como_llegar", ROUTE_DESCRIPTION, RouteArgs, self._route),
            )
        }

    @property
    def index(self) -> GuideIndex:
        """Índice de la guía, creado al primer uso."""
        if self._index is None:
            self._index = build_index()
        return self._index

    def declarations(self) -> list[dict[str, Any]]:
        """Declaraciones de todas las herramientas en JSON Schema."""
        return [tool.declaration() for tool in self.tools.values()]

    def gemini_tools(self) -> list[types.Tool]:
        """Declaraciones en el formato del SDK de Gemini."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=declaration["name"],
                        description=declaration["description"],
                        parameters_json_schema=declaration["parameters"],
                    )
                    for declaration in self.declarations()
                ]
            )
        ]

    def execute(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Ejecuta una herramienta pedida por el LLM sin lanzar excepciones.

        Args:
            name: Nombre de la herramienta.
            args: Argumentos que ha generado el LLM.

        Returns:
            Resultado de la herramienta. Ante cualquier fallo, un diccionario
            con ``"ok": False``, el tipo de ``error`` y un ``mensaje``.
        """
        args = dict(args or {})
        start = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            result = {
                "ok": False,
                "error": "herramienta_desconocida",
                "mensaje": f"No existe la herramienta {name!r}. "
                f"Disponibles: {', '.join(self.tools)}.",
            }
        else:
            try:
                result = tool.run(args)
            except Exception:  # Último recurso: la conversación debe continuar.
                logger.exception("Fallo inesperado en la herramienta %s", name)
                result = {
                    "ok": False,
                    "error": "error_interno",
                    "mensaje": "La herramienta ha fallado de forma inesperada. "
                    "Responde sin ella o propón intentarlo de nuevo.",
                }
        log_event(
            name,
            stage="llamada",
            args=args,
            ok=result.get("ok", False),
            error=result.get("error"),
            result=_log_summary(result),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
        return result

    def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        """``buscar_guia``: búsqueda semántica en la guía."""
        try:
            parsed = SearchArgs.model_validate(args)
        except ValidationError as err:
            return _invalid_args(err)
        zona = None if parsed.zona == "toda" else parsed.zona
        try:
            results = self.index.search(parsed.consulta, zona=zona)
        except LLMError as err:
            return {"ok": False, "error": "servicio_no_disponible", "mensaje": str(err)}
        response: dict[str, Any] = {
            "ok": True,
            "consulta": parsed.consulta,
            "zona": parsed.zona,
            "fragmentos": [
                {"fuente": result.citation, "texto": result.text} for result in results
            ],
        }
        if not results:
            response["mensaje"] = (
                "La guía no tiene información relevante para esta consulta."
            )
        return response

    def _weather(self, args: dict[str, Any]) -> dict[str, Any]:
        """``get_weather``: delega en :func:`src.weather.get_weather`.

        La validación, los reintentos y el registro de cada intento los hace
        el propio módulo del tiempo.
        """
        return weather.get_weather(args.get("fecha", ""), args.get("lugar", ""))

    def _route(self, args: dict[str, Any]) -> dict[str, Any]:
        """``como_llegar``: enlace de ruta de Google Maps hasta un sitio."""
        try:
            parsed = RouteArgs.model_validate(args)
        except ValidationError as err:
            return _invalid_args(err)
        destination = parsed.sitio.strip()
        if "tenerife" not in destination.lower():
            destination += ", Tenerife"
        return {
            "ok": True,
            "sitio": parsed.sitio,
            "url": MAPS_DIRECTIONS_URL + quote_plus(destination),
            "nota": "Ruta desde la ubicación actual del usuario.",
        }
