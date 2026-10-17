"""Tests de la app web: panel de observabilidad, límites y streaming."""

from dataclasses import replace

import app
from src.agent import ToolCallTrace, TurnTrace
from src.trace_view import (
    SessionStats,
    call_outcome,
    format_args,
    sources_markdown,
    trace_html,
)


def search_call(*sources: str) -> ToolCallTrace:
    fragments = [{"fuente": s, "texto": "Texto del fragmento " * 30} for s in sources]
    return ToolCallTrace(
        name="buscar_guia",
        args={"consulta": "Masca", "zona": "norte"},
        ok=True,
        error=None,
        latency_ms=1234,
        sources=list(sources),
        result={"ok": True, "fragmentos": fragments},
    )


def weather_call(ok: bool = True) -> ToolCallTrace:
    result = (
        {
            "ok": True,
            "lugar": "Masca",
            "fecha": "2026-10-18",
            "cielo": "Despejado",
            "temperatura_min_c": 16,
            "temperatura_max_c": 24,
        }
        if ok
        else {"ok": False, "error": "fecha_pasada"}
    )
    return ToolCallTrace(
        name="get_weather",
        args={"fecha": "2026-10-18", "lugar": "Masca"},
        ok=ok,
        error=None if ok else "fecha_pasada",
        latency_ms=300,
        result=result,
    )


def test_format_args_and_outcomes():
    assert format_args({"lugar": "Masca", "fecha": "2026-10-18"}) == (
        'lugar="Masca", fecha="2026-10-18"'
    )
    assert call_outcome(search_call("p. 12", "p. 13")) == "2 fragmentos"
    assert call_outcome(weather_call()) == "Despejado, 16–24 °C"
    assert call_outcome(weather_call(ok=False)) == "Error: fecha_pasada"


def test_trace_html_shows_tools_tokens_and_latency():
    trace = TurnTrace(
        rounds=2,
        tool_calls=[search_call("p. 12")],
        prompt_tokens=12345,
        output_tokens=280,
        last_prompt_tokens=3100,
        first_token_s=1.24,
        total_s=6.8,
        model="gemini-3.5-flash-lite",
    )
    text = trace_html(trace)
    assert "buscar_guia(consulta=&quot;Masca&quot;, zona=&quot;norte&quot;)" in text
    assert "1.234 ms" in text
    assert "12.345 entrada" in text
    assert "3.100 / 6.000 tokens" in text
    assert "width:52%" in text
    assert "1,2 s" in text and "6,8 s" in text


def test_trace_html_without_tools_escapes_error():
    text = trace_html(TurnTrace(error="Fallo <b>"))
    assert "sin usar herramientas" in text
    assert "Fallo &lt;b&gt;" in text


def test_sources_dedupe_truncate_and_skip_failures():
    trace = TurnTrace(
        tool_calls=[
            search_call("p. 12 · Masca"),
            search_call("p. 12 · Masca"),
            weather_call(),
            weather_call(ok=False),
        ]
    )
    text = sources_markdown(trace, "Masca está en Teno (TENERIFE.pdf, p. 12).")
    assert text.count("p. 12 · Masca") == 1
    assert "…" in text
    assert "Open-Meteo** · Masca, 2026-10-18" in text
    assert sources_markdown(TurnTrace()) is None


def test_sources_separate_cited_from_unused():
    trace = TurnTrace(
        tool_calls=[search_call("TENERIFE.pdf, p. 22 · Masca", "p. 16-17 · Adeje")]
    )
    text = sources_markdown(trace, "Visita Masca (TENERIFE.pdf, p. 22).")
    cited, unused = text.split("También consultadas")
    assert "**✓ TENERIFE.pdf, p. 22 · Masca**" in cited
    assert "Adeje" not in cited
    assert "- p. 16-17 · Adeje" in unused
    # Sin citas en la respuesta, todo lo recuperado queda como no usado.
    assert "Citadas" not in sources_markdown(trace, "Sin citas.")


def test_session_stats_accumulate():
    stats = SessionStats()
    stats.add(
        TurnTrace(prompt_tokens=100, output_tokens=10, tool_calls=[weather_call()])
    )
    stats.add(TurnTrace(prompt_tokens=50, output_tokens=5))
    assert (stats.turns, stats.prompt_tokens, stats.tool_calls) == (2, 150, 1)
    assert "2 / 20" in stats.html(20)


class FakeAgent:
    """Agente que responde en dos trozos con una búsqueda en la guía."""

    def __init__(self):
        self.last_trace = TurnTrace()

    def chat_stream(self, message):
        self.last_trace = TurnTrace(tool_calls=[search_call("p. 23 · Masca")])
        yield "Masca está "
        yield "en Teno (p. 23)."


def test_paced_splits_words_and_keeps_text():
    chunks = ["Masca está ", "en Teno\n\n(p. 23)."]
    words = list(app.paced(iter(chunks), delay_s=0))
    assert "".join(words) == "".join(chunks)
    assert words[:3] == ["Masca ", "está ", "en "]


def test_respond_streams_answer_and_appends_sources(monkeypatch):
    monkeypatch.setattr(app, "APP", replace(app.APP, word_delay_s=0))
    session = app.Session(FakeAgent())
    steps = list(app.respond("¿Qué es Masca?", [], session))
    history, textbox, returned, panel = steps[-1]
    assert history[0] == {"role": "user", "content": "¿Qué es Masca?"}
    assert history[1]["content"] == "Masca está en Teno (p. 23)."
    assert history[2]["metadata"]["title"] == "📚 Fuentes consultadas"
    assert textbox == "" and returned is session
    assert "Búsqueda en la guía" in panel
    assert len(steps) == 8  # "Pensando", seis palabras y el final.
    assert session.stats.turns == 1


def test_respond_enforces_limits(monkeypatch):
    session = app.Session(FakeAgent())
    session.stats.turns = app.APP.max_messages
    history, *_ = list(app.respond("Hola", [], session))[-1]
    assert "límite" in history[-1]["content"]

    monkeypatch.setattr(app, "APP", replace(app.APP, max_message_chars=5))
    history, *_ = list(app.respond("Mensaje largo", [], app.Session(FakeAgent())))[-1]
    assert "demasiado largo" in history[-1]["content"]


def test_respond_ignores_empty_message():
    steps = list(app.respond("   ", [], None))
    assert steps[0][0] == []
