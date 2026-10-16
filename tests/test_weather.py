"""Tests de get_weather sin red: la API de Open-Meteo se sustituye por un falso."""

from datetime import date

import pytest
import requests

from src import tool_log, weather
from src.weather import DAILY_FIELDS, PLACES, get_weather, tool_declaration

TODAY = date(2026, 10, 16)


def payload(day="2026-10-17", code=45):
    """Respuesta mínima de Open-Meteo con la forma real."""
    daily = {
        "time": [day],
        "weather_code": [code],
        **{name: [1.5] for name in DAILY_FIELDS},
        "sunrise": [f"{day}T08:07"],
        "sunset": [f"{day}T19:34"],
    }
    return {"elevation": 520.6, "daily": daily}


class FakeResponse:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self.ok = status < 400
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError("no es JSON")
        return self._data


class FakeAPI:
    """Devuelve, por orden, las respuestas o excepciones programadas."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, params, timeout):
        self.calls.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fija "hoy", redirige el registro y elimina las esperas entre intentos."""
    monkeypatch.setattr(weather, "today", lambda config=None: TODAY)
    monkeypatch.setattr(tool_log, "LOG_PATH", tmp_path / "tools.jsonl")
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: None)


@pytest.fixture()
def api(monkeypatch):
    def install(*outcomes):
        fake = FakeAPI(*outcomes)
        monkeypatch.setattr(weather.requests, "get", fake)
        return fake

    return install


# --- Llamadas correctas -----------------------------------------------------


def test_valid_call_returns_tourist_fields(api):
    fake = api(FakeResponse(data=payload()))
    result = get_weather("2026-10-17", "Masca")

    assert result["ok"] is True
    assert result["lugar"] == "Masca"
    assert result["cielo"] == "Niebla"
    assert result["altitud_m"] == 521
    assert result["amanecer"] == "08:07"
    assert result["atardecer"] == "19:34"
    assert set(DAILY_FIELDS.values()) <= result.keys()
    assert "aviso" not in result
    assert fake.calls[0]["start_date"] == "2026-10-17"
    assert fake.calls[0]["latitude"] == PLACES["Masca"][0]

    log = tool_log.read_log()
    assert len(log) == 1
    assert log[0]["ok"] is True
    assert log[0]["attempt"] == 1


@pytest.mark.parametrize(
    ("written", "canonical"),
    [("masca", "Masca"), ("EL MEDANO", "El Médano"), ("la  orotava", "La Orotava")],
)
def test_place_names_are_normalized(api, written, canonical):
    api(FakeResponse(data=payload()))
    assert get_weather("2026-10-17", written)["lugar"] == canonical


def test_forecast_range_limits_are_accepted(api):
    api(FakeResponse(data=payload()), FakeResponse(data=payload()))
    assert get_weather("2026-10-16", "Masca")["ok"] is True  # hoy
    far = get_weather("2026-10-31", "Masca")  # hoy + 15
    assert far["ok"] is True
    assert "orientativo" in far["aviso"]


# --- Errores de validación (no llegan a la API) -----------------------------


@pytest.mark.parametrize(
    ("fecha", "lugar", "kind"),
    [
        ("17/10/2026", "Masca", "formato_fecha"),
        ("sábado", "Masca", "formato_fecha"),
        ("2026-10-15", "Masca", "fecha_pasada"),
        ("2026-11-15", "Masca", "fecha_lejana"),
        ("2026-10-17", "Mazca", "lugar_desconocido"),
        ("2026-10-17", "Gran Canaria", "lugar_desconocido"),
    ],
)
def test_invalid_arguments_never_call_the_api(api, fecha, lugar, kind):
    fake = api()
    result = get_weather(fecha, lugar)

    assert result["ok"] is False
    assert result["error"] == kind
    assert result["mensaje"]
    assert fake.calls == []
    assert tool_log.read_log()[0]["stage"] == "validacion"


def test_far_date_says_when_it_will_be_available(api):
    api()
    result = get_weather("2026-11-15", "Masca")
    assert "2026-10-31" in result["mensaje"]


def test_unknown_place_lists_valid_places(api):
    api()
    result = get_weather("2026-10-17", "Mazca")
    assert "Masca" in result["lugares_validos"]


def test_all_wrong_arguments_are_reported(api):
    api()
    result = get_weather("mañana", "Mazca")
    assert result["error"] == "formato_fecha"
    assert "Mazca" in result["mensaje"]


# --- Fallos de la API ---------------------------------------------------------


def test_transient_failure_is_retried(api):
    fake = api(requests.Timeout(), FakeResponse(data=payload()))
    result = get_weather("2026-10-17", "Masca")

    assert result["ok"] is True
    assert len(fake.calls) == 2
    log = tool_log.read_log()
    assert [(e["attempt"], e["ok"]) for e in log] == [(1, False), (2, True)]
    assert log[0]["transient"] is True


def test_persistent_outage_returns_clear_error(api):
    fake = api(FakeResponse(503), FakeResponse(503))
    result = get_weather("2026-10-17", "Masca")

    assert result == {
        "ok": False,
        "error": "servicio_no_disponible",
        "mensaje": result["mensaje"],
    }
    assert "HTTP 503" in result["mensaje"]
    assert len(fake.calls) == 2


def test_client_error_is_not_retried(api):
    fake = api(FakeResponse(400, {"error": True}))
    result = get_weather("2026-10-17", "Masca")
    assert result["error"] == "respuesta_invalida"
    assert len(fake.calls) == 1


@pytest.mark.parametrize("data", [None, {}, {"daily": {"weather_code": []}}])
def test_malformed_response_is_handled(api, data):
    api(FakeResponse(data=data))
    assert get_weather("2026-10-17", "Masca")["error"] == "respuesta_invalida"


# --- Esquema para el LLM ----------------------------------------------------


def test_tool_declaration_is_clean_json_schema():
    declaration = tool_declaration()
    params = declaration["parameters"]

    assert declaration["name"] == "get_weather"
    assert params["required"] == ["fecha", "lugar"]
    assert params["properties"]["fecha"]["format"] == "date"
    assert params["properties"]["lugar"]["enum"] == list(PLACES)
    assert "description" not in params  # sin el docstring para desarrolladores
