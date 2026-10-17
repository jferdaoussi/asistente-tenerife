"""Presentación de la traza de un turno para la app web.

Convierte un :class:`~src.agent.TurnTrace` en el HTML del panel de
observabilidad (latencia, tokens, memoria y herramientas) y en el Markdown de
las fuentes consultadas. No depende de Gradio, así que se prueba con pytest.
Los estilos (clases ``obs-*``) están en ``app.py``.
"""

from dataclasses import dataclass
from html import escape
from typing import Any

from src.agent import ToolCallTrace, TurnTrace
from src.chunking import cited_pages
from src.config import MEMORY

#: Caracteres de cada fragmento que se muestran en las fuentes.
SNIPPET_CHARS = 300

#: Nombre legible de cada herramienta en el panel.
TOOL_LABELS = {
    "buscar_guia": "Búsqueda en la guía",
    "get_weather": "Pronóstico del tiempo",
    "como_llegar": "Ruta en Google Maps",
}


def _int(value: int) -> str:
    """Entero con separador de miles español: 12345 → "12.345"."""
    return f"{value:,}".replace(",", ".")


def _seconds(value: float | None) -> str:
    """Segundos con coma decimal: 1.25 → "1,2 s"."""
    return "—" if value is None else f"{value:.1f} s".replace(".", ",")


def format_args(args: dict[str, Any]) -> str:
    """Argumentos de una llamada en una línea: ``lugar="Masca", fecha=...``."""
    return ", ".join(f'{name}="{value}"' for name, value in args.items())


def call_outcome(call: ToolCallTrace) -> str:
    """Resultado de una llamada a herramienta, en pocas palabras."""
    if not call.ok:
        return f"Error: {call.error}"
    result = call.result
    if call.name == "buscar_guia":
        found = len(result.get("fragmentos", []))
        return f"{found} fragmentos" if found else "Sin fragmentos relevantes"
    if call.name == "get_weather":
        return (
            f"{result.get('cielo', '')}, "
            f"{result.get('temperatura_min_c')}–{result.get('temperatura_max_c')} °C"
        )
    if call.name == "como_llegar":
        return "Enlace generado"
    return "Correcto"


def _tile(label: str, value: str, detail: str) -> str:
    """Tarjeta de una métrica."""
    return (
        f'<div class="obs-tile"><span class="obs-label">{label}</span>'
        f'<span class="obs-value">{value}</span>'
        f'<span class="obs-detail">{detail}</span></div>'
    )


def _tool_item(call: ToolCallTrace) -> str:
    """Una llamada a herramienta en la línea de tiempo del turno."""
    status = "ok" if call.ok else "error"
    label = TOOL_LABELS.get(call.name, call.name)
    return (
        f'<li class="obs-tool obs-{status}">'
        f'<div class="obs-tool-head"><span>{escape(label)}</span>'
        f'<span class="obs-ms">{_int(call.latency_ms)} ms</span></div>'
        f"<code>{escape(call.name)}({escape(format_args(call.args))})</code>"
        f'<div class="obs-outcome">{escape(call_outcome(call))}</div></li>'
    )


def trace_html(trace: TurnTrace) -> str:
    """Panel de observabilidad de un turno.

    Args:
        trace: Traza del turno que acaba de terminar.

    Returns:
        HTML con métricas, uso de la memoria y herramientas ejecutadas.
    """
    tiles = "".join(
        (
            _tile(
                "Latencia",
                _seconds(trace.total_s),
                f"1.ª palabra: {_seconds(trace.first_token_s)}",
            ),
            _tile(
                "Tokens",
                _int(trace.prompt_tokens + trace.output_tokens),
                f"{_int(trace.prompt_tokens)} entrada · "
                f"{_int(trace.output_tokens)} salida",
            ),
            _tile("Vueltas", str(trace.rounds), "del bucle del agente"),
            _tile("Herramientas", str(len(trace.tool_calls)), "llamadas"),
        )
    )

    budget = MEMORY.max_history_tokens
    used = min(trace.last_prompt_tokens / budget, 1.0) if budget else 0.0
    memory_note = (
        "Turnos antiguos resumidos."
        if trace.memory_compacted
        else "Al superar el presupuesto se resumen los turnos antiguos."
    )
    memory = (
        '<div class="obs-section"><div class="obs-row">'
        "<span>Memoria de la conversación</span>"
        f"<span>{_int(trace.last_prompt_tokens)} / {_int(budget)} tokens</span>"
        '</div><div class="obs-bar">'
        f'<div class="obs-fill" style="width:{used:.0%}"></div></div>'
        f'<span class="obs-detail">{memory_note}</span></div>'
    )

    if trace.tool_calls:
        tools = "".join(_tool_item(call) for call in trace.tool_calls)
        tools = f'<ol class="obs-tools">{tools}</ol>'
    else:
        tools = '<p class="obs-detail">Respondió sin usar herramientas.</p>'
    model = (
        f'<p class="obs-detail">Modelo: <code>{escape(trace.model or "—")}</code>'
        + (" · respuesta forzada" if trace.forced_final else "")
        + "</p>"
    )
    error = f'<p class="obs-alert">{escape(trace.error)}</p>' if trace.error else ""
    return (
        f'<div class="obs-grid">{tiles}</div>{memory}'
        '<div class="obs-section"><span class="obs-heading">'
        f"Herramientas del turno</span>{tools}{model}</div>{error}"
    )


def sources_markdown(trace: TurnTrace, answer: str = "") -> str | None:
    """Fuentes del turno, separando las citadas en la respuesta.

    Un fragmento de la guía cuenta como citado si alguna de sus páginas
    aparece citada en la respuesta. Los citados se muestran con su texto; los
    que se recuperaron pero no se usaron, solo con su título, para que se vea
    qué encontró la búsqueda sin confundirlos con la base de la respuesta.

    Args:
        trace: Traza del turno.
        answer: Texto de la respuesta del asistente.

    Returns:
        Texto Markdown, o None si el turno no consultó ninguna fuente.
    """
    answer_pages = cited_pages(answer)
    cited: list[str] = []
    unused: list[str] = []
    external: list[str] = []
    seen: set[str] = set()
    for call in trace.tool_calls:
        if not call.ok:
            continue
        if call.name == "buscar_guia":
            for fragment in call.result.get("fragmentos", []):
                source = fragment["fuente"]
                if source in seen:
                    continue
                seen.add(source)
                if not cited_pages(source) & answer_pages:
                    unused.append(f"- {source}")
                    continue
                text = fragment["texto"]
                if len(text) > SNIPPET_CHARS:
                    text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
                cited.append(f"**✓ {source}**\n\n{text}")
        elif call.name == "get_weather":
            external.append(
                f"**☀️ Open-Meteo** · {call.result.get('lugar')}, "
                f"{call.result.get('fecha')}"
            )
        elif call.name == "como_llegar":
            external.append(
                f"**🧭 Ruta** · [{call.result.get('sitio')}]({call.result.get('url')})"
            )

    sections = []
    if cited:
        sections.append("#### Citadas en la respuesta\n\n" + "\n\n".join(cited))
    if external:
        sections.append("#### Datos externos\n\n" + "\n\n".join(external))
    if unused:
        sections.append(
            "#### También consultadas (no usadas en la respuesta)\n\n"
            + "\n".join(unused)
        )
    return "\n\n---\n\n".join(sections) if sections else None


@dataclass
class SessionStats:
    """Totales de una conversación en la app."""

    turns: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    def add(self, trace: TurnTrace) -> None:
        """Suma un turno terminado."""
        self.turns += 1
        self.prompt_tokens += trace.prompt_tokens
        self.output_tokens += trace.output_tokens
        self.tool_calls += len(trace.tool_calls)

    def html(self, max_messages: int) -> str:
        """Pie del panel con los totales de la conversación."""
        return (
            '<div class="obs-footer">'
            f"<span>Mensajes <b>{self.turns} / {max_messages}</b></span>"
            f"<span>Herramientas <b>{self.tool_calls}</b></span>"
            f"<span>Tokens <b>{_int(self.prompt_tokens + self.output_tokens)}</b>"
            "</span></div>"
        )
