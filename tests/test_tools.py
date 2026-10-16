"""Tests del registro de herramientas sin red ni API."""

from dataclasses import replace
from urllib.parse import unquote_plus

import pytest
from google.genai import types

from src import tool_log, weather
from src.chunking import Chunk
from src.llm import LLMError
from src.rag import GuideIndex
from src.tools import ToolRegistry
from tests.fakes import FakeEmbedder


def _chunk(chunk_id, text, zona, page):
    return Chunk(
        chunk_id, f"Zona › {chunk_id}\n{text}", chunk_id, zona, "lugar", page, page
    )


@pytest.fixture(autouse=True)
def isolated_log(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_log, "LOG_PATH", tmp_path / "tools.jsonl")


@pytest.fixture()
def registry(tmp_path):
    index = GuideIndex(embedder=FakeEmbedder(), persist_dir=tmp_path / "chroma")
    index.build(
        [
            _chunk("Benijo", "Playa de arena negra en el norte", "norte", 8),
            _chunk("Teide", "Subir al Teide y ver el Teide nevado", "norte", 15),
            _chunk("Adeje", "Fiesta y playa en el sur", "sur", 17),
        ]
    )
    return ToolRegistry(index=index)


# --- Declaraciones ----------------------------------------------------------


def test_declarations_are_clean_json_schema(registry):
    declarations = {d["name"]: d for d in registry.declarations()}
    assert set(declarations) == {"buscar_guia", "get_weather", "como_llegar"}

    for declaration in declarations.values():
        params = declaration["parameters"]
        assert declaration["description"]
        assert params["type"] == "object"
        assert "title" not in params and "description" not in params
        assert all("title" not in prop for prop in params["properties"].values())

    search = declarations["buscar_guia"]["parameters"]
    assert search["required"] == ["consulta"]
    assert search["properties"]["zona"]["enum"] == ["norte", "sur", "toda"]

    weather_params = declarations["get_weather"]["parameters"]
    assert weather_params["required"] == ["fecha", "lugar"]
    assert weather_params["properties"]["fecha"]["format"] == "date"
    assert weather_params["properties"]["lugar"]["enum"] == list(weather.PLACES)


def test_gemini_tools_wrap_all_declarations(registry):
    (tool,) = registry.gemini_tools()
    assert isinstance(tool, types.Tool)
    names = [fd.name for fd in tool.function_declarations]
    assert names == ["buscar_guia", "get_weather", "como_llegar"]


# --- buscar_guia ------------------------------------------------------------


def test_search_returns_fragments_with_sources(registry):
    result = registry.execute("buscar_guia", {"consulta": "subir al teide"})
    assert result["ok"] is True
    first = result["fragmentos"][0]
    assert first["fuente"] == "TENERIFE.pdf, p. 15 · Teide"
    assert "Teide nevado" in first["texto"]


def test_search_applies_zone_filter(registry):
    result = registry.execute(
        "buscar_guia", {"consulta": "playa en el sur", "zona": "sur"}
    )
    assert [f["fuente"] for f in result["fragmentos"]] == [
        "TENERIFE.pdf, p. 17 · Adeje"
    ]


def test_search_without_results_says_so(registry, monkeypatch):
    strict = replace(registry.index.config, min_similarity=0.95)
    monkeypatch.setattr(registry.index, "config", strict)
    result = registry.execute("buscar_guia", {"consulta": "museos de arte moderno"})
    assert result["ok"] is True
    assert result["fragmentos"] == []
    assert "no tiene información" in result["mensaje"]


@pytest.mark.parametrize(
    "args", [{}, {"consulta": ""}, {"consulta": "playas", "zona": "este"}]
)
def test_search_rejects_invalid_arguments(registry, args):
    result = registry.execute("buscar_guia", args)
    assert result["error"] == "argumentos_invalidos"


def test_search_reports_embedding_outage(registry, monkeypatch):
    def broken(query):
        raise LLMError("Se ha alcanzado el límite de uso gratuito.")

    monkeypatch.setattr(registry.index.embedder, "embed_query", broken)
    result = registry.execute("buscar_guia", {"consulta": "playas"})
    assert result["error"] == "servicio_no_disponible"


# --- get_weather y como_llegar ------------------------------------------------


def test_weather_is_delegated_to_weather_module(registry, monkeypatch):
    calls = []

    def fake_get_weather(fecha, lugar):
        calls.append((fecha, lugar))
        return {"ok": True}

    monkeypatch.setattr(weather, "get_weather", fake_get_weather)
    registry.execute("get_weather", {"fecha": "2026-10-17", "lugar": "Masca"})
    registry.execute("get_weather", {})
    assert calls == [("2026-10-17", "Masca"), ("", "")]


def test_route_builds_google_maps_link(registry):
    result = registry.execute("como_llegar", {"sitio": "Playa de Benijo"})
    assert result["ok"] is True
    assert result["url"].startswith("https://www.google.com/maps/dir/?api=1")
    assert unquote_plus(result["url"]).endswith("destination=Playa de Benijo, Tenerife")


def test_route_does_not_duplicate_island_name(registry):
    result = registry.execute("como_llegar", {"sitio": "Teide, Tenerife"})
    assert unquote_plus(result["url"]).endswith("destination=Teide, Tenerife")


# --- Ejecución segura y registro ----------------------------------------------


def test_unknown_tool_is_reported_not_raised(registry):
    result = registry.execute("reservar_hotel", {"sitio": "Masca"})
    assert result["error"] == "herramienta_desconocida"
    assert "buscar_guia" in result["mensaje"]


def test_unexpected_exception_is_contained(registry, monkeypatch):
    def explode(query, **kwargs):
        raise RuntimeError("fallo imprevisto")

    monkeypatch.setattr(registry.index, "search", explode)
    result = registry.execute("buscar_guia", {"consulta": "playas"})
    assert result["error"] == "error_interno"


def test_every_call_is_logged_compactly(registry):
    registry.execute("buscar_guia", {"consulta": "subir al teide"})
    registry.execute("reservar_hotel", None)

    search, unknown = tool_log.read_log()
    assert search["stage"] == "llamada"
    assert search["ok"] is True
    assert search["result"]["fragmentos"][0] == "TENERIFE.pdf, p. 15 · Teide"
    assert unknown["ok"] is False
    assert unknown["error"] == "herramienta_desconocida"
