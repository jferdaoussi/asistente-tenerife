"""App web del asistente turístico de Tenerife (Gradio).

Capa fina sobre el agente: el chat muestra la respuesta en streaming y las
fuentes consultadas, y el panel lateral, la traza de cada turno. Cada
visitante tiene su propio agente (su memoria); el índice de la guía se
comparte.

Uso local: ``python app.py`` y abrir http://127.0.0.1:7860.
"""

import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

from src.agent import TouristAgent
from src.config import APP
from src.llm import GeminiClient
from src.rag import build_index
from src.tools import ToolRegistry
from src.trace_view import SessionStats, sources_markdown, trace_html

TITLE = "Asistente turístico de Tenerife"
HEADER = """
<div class="hero">
  <div class="hero-eyebrow">Guía inteligente · Tenerife</div>
  <h1>Asistente turístico de Tenerife</h1>
  <p>Pregunta por sitios, planes, rutas o el tiempo. Cada dato de la guía
  llega con su página, y el pronóstico, en directo.</p>
  <div class="hero-chips">
    <span>📖 Guía <i>TENERIFE.pdf</i> con citas</span>
    <span>☀️ Pronóstico de Open-Meteo</span>
    <span>🧭 Rutas en Google Maps</span>
    <span>💬 Recuerda la conversación</span>
  </div>
</div>
"""
EMPTY_PANEL = (
    '<div class="obs-empty"><b>Aún no hay turnos.</b><br>Haz una pregunta para '
    "ver cómo trabaja el agente: herramientas que usa, tokens, latencia y "
    "memoria.</div>"
)
LOADING_PANEL = (
    '<div class="obs-loading"><span class="obs-pulse"></span>'
    "El agente está trabajando…</div>"
)
EXAMPLES = [
    "¿Qué puedo ver en La Laguna?",
    "¿Qué tiempo hará mañana en Masca?",
    "¿Cómo llego al Teide?",
    "¿Dónde puedo comprar gofio?",
    "¿Qué me recomiendas en Gran Canaria?",
]
FOOTER = (
    f"Demo con límites: {APP.max_messages} mensajes por conversación y "
    f"{APP.max_message_chars} caracteres por mensaje · Gemini Flash-Lite · "
    "Proyecto final de <i>Large Language Models</i> (Pontia)"
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }
.hero {
  padding: 28px 32px; border-radius: 18px; color: #f8fafc;
  background: radial-gradient(circle at 85% 0%, rgba(251,146,60,.35), transparent 45%),
              linear-gradient(135deg, #0b1f3a 0%, #0f3b5f 55%, #135e74 100%);
  box-shadow: 0 10px 30px rgba(11,31,58,.25);
}
.hero h1 { margin: 6px 0 8px; font-size: 30px; font-weight: 700;
  letter-spacing: -.02em; color: #fff; }
.hero p { margin: 0; max-width: 680px; opacity: .85; font-size: 15px; }
.hero-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: #fdba74; }
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.hero-chips span { font-size: 13px; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18); }
.card { border: 1px solid var(--border-color-primary) !important;
  border-radius: 16px !important; background: var(--block-background-fill) !important;
  box-shadow: 0 1px 3px rgba(15,23,42,.06); padding: 14px !important; }
#chat { border: none !important; box-shadow: none !important; }
#chat h4 { font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: .08em; text-transform: uppercase; opacity: .75;
  margin: 6px 0 8px !important; }
#composer { align-items: stretch; gap: 10px; }
#composer textarea { font-size: 15px !important; }
.chips { flex-wrap: wrap !important; gap: 8px !important; }
.chip { flex: 0 0 auto !important; min-width: 0 !important;
  border-radius: 999px !important; font-weight: 500 !important;
  font-size: 13px !important; padding: 6px 14px !important;
  background: var(--background-fill-secondary) !important;
  border: 1px solid var(--border-color-primary) !important; }
.chip:hover { border-color: var(--color-accent) !important;
  color: var(--color-accent) !important; }
.panel-title { font-size: 13px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--body-text-color-subdued);
  margin: 2px 2px 10px; }
.obs-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.obs-tile { display: flex; flex-direction: column; gap: 2px; padding: 12px 14px;
  border-radius: 12px; background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary); }
.obs-label { font-size: 12px; color: var(--body-text-color-subdued); }
.obs-value { font-size: 22px; font-weight: 700; letter-spacing: -.01em; }
.obs-detail { font-size: 12px; color: var(--body-text-color-subdued); }
.obs-section { margin-top: 16px; display: flex; flex-direction: column; gap: 6px; }
.obs-heading { font-size: 13px; font-weight: 600; }
.obs-row { display: flex; justify-content: space-between; font-size: 13px;
  font-weight: 600; }
.obs-bar { height: 8px; border-radius: 999px;
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary); overflow: hidden; }
.obs-fill { height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #0ea5e9, #f97316); }
.obs-tools { list-style: none; margin: 4px 0 0; padding: 0 0 0 14px;
  border-left: 2px solid var(--border-color-primary); display: flex;
  flex-direction: column; gap: 12px; }
.obs-tool { position: relative; font-size: 13px; }
.obs-tool::before { content: ""; position: absolute; left: -21px; top: 4px; width: 10px;
  height: 10px; border-radius: 50%; background: #10b981;
  box-shadow: 0 0 0 3px var(--block-background-fill); }
.obs-tool.obs-error::before { background: #ef4444; }
.obs-tool-head { display: flex; justify-content: space-between; font-weight: 600; }
.obs-ms { font-weight: 500; color: var(--body-text-color-subdued); }
.obs-tool code { display: block; margin: 4px 0; font-size: 12px; white-space: normal;
  word-break: break-word; }
.obs-outcome { color: var(--body-text-color-subdued); }
.obs-error .obs-outcome { color: #ef4444; }
.obs-alert { margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 13px;
  background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.35); }
.obs-footer { margin-top: 16px; padding-top: 12px; display: flex; gap: 14px;
  flex-wrap: wrap; font-size: 12px; color: var(--body-text-color-subdued);
  border-top: 1px solid var(--border-color-primary); }
.obs-footer b { color: var(--body-text-color); }
.obs-empty, .obs-loading { padding: 18px; border-radius: 12px; font-size: 14px;
  line-height: 1.5; color: var(--body-text-color-subdued);
  background: var(--background-fill-secondary);
  border: 1px dashed var(--border-color-primary); }
.obs-loading { display: flex; align-items: center; gap: 10px; }
.obs-pulse { width: 10px; height: 10px; border-radius: 50%; background: #f97316;
  animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .3; transform: scale(.8); } }
.app-footer { text-align: center; font-size: 12px;
  color: var(--body-text-color-subdued); padding: 6px 0 12px; }
"""

THEME = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
)

Message = dict[str, Any]

_tools: ToolRegistry | None = None
_tools_lock = threading.Lock()


def shared_tools() -> ToolRegistry:
    """Herramientas con el índice de la guía, creadas una sola vez."""
    global _tools
    with _tools_lock:
        if _tools is None:
            _tools = ToolRegistry(build_index())
        return _tools


@dataclass
class Session:
    """Conversación de un visitante: su agente y sus totales."""

    agent: TouristAgent
    stats: SessionStats = field(default_factory=SessionStats)

    @classmethod
    def start(cls) -> "Session":
        """Crea el agente de un visitante nuevo."""
        return cls(TouristAgent(client=GeminiClient(), tools=shared_tools()))


def paced(chunks: Iterator[str], delay_s: float) -> Iterator[str]:
    """Reparte cada trozo del streaming en palabras, con una pausa entre ellas.

    Args:
        chunks: Trozos de texto tal como llegan del modelo.
        delay_s: Pausa tras cada palabra, en segundos.

    Yields:
        Palabras con su espacio final; juntas reproducen el texto exacto.
    """
    for chunk in chunks:
        for word in re.findall(r"\s*\S+\s*|\s+", chunk):
            yield word
            time.sleep(delay_s)


def limit_message(message: str, session: Session | None) -> str | None:
    """Motivo por el que no se atiende un mensaje, o None si se atiende."""
    if len(message) > APP.max_message_chars:
        return (
            f"El mensaje es demasiado largo (máximo {APP.max_message_chars} "
            "caracteres). ¿Puedes resumirlo?"
        )
    if session and session.stats.turns >= APP.max_messages:
        return (
            f"Has llegado al límite de {APP.max_messages} mensajes de esta "
            "demo. Pulsa «Nueva conversación» para empezar otra."
        )
    return None


def respond(
    message: str, history: list[Message], session: Session | None
) -> Iterator[tuple[list[Message], str, Session | None, Any]]:
    """Atiende un mensaje y va emitiendo la respuesta en streaming.

    Yields:
        Historial del chat, texto de la caja (vacía), sesión y panel.
    """
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        yield history, "", session, gr.skip()
        return

    history.append({"role": "user", "content": message})
    refusal = limit_message(message, session)
    if session is None and refusal is None:
        try:
            session = Session.start()
        except Exception as err:  # Sin clave o sin acceso a la API.
            refusal = f"⚠️ El asistente no está disponible: {err}"
    if refusal:
        history.append({"role": "assistant", "content": refusal})
        yield history, "", session, gr.skip()
        return

    history.append({"role": "assistant", "content": ""})
    yield history, "", session, LOADING_PANEL
    for chunk in paced(session.agent.chat_stream(message), APP.word_delay_s):
        history[-1]["content"] += chunk
        yield history, "", session, gr.skip()

    trace = session.agent.last_trace
    session.stats.add(trace)
    sources = sources_markdown(trace, history[-1]["content"])
    if sources:
        history.append(
            {
                "role": "assistant",
                "content": sources,
                "metadata": {"title": "📚 Fuentes consultadas", "status": "done"},
            }
        )
    yield history, "", session, trace_html(trace) + session.stats.html(APP.max_messages)


def credentials() -> tuple[str, str] | None:
    """Usuario y contraseña de acceso, si están definidos en el entorno.

    En el despliegue se definen ``APP_USER`` y ``APP_PASSWORD`` como secretos
    para que solo entre quien los conozca (y no se gaste la cuota). En local,
    sin ellos, la app abre sin pedir acceso.
    """
    user = os.getenv("APP_USER", "").strip()
    password = os.getenv("APP_PASSWORD", "").strip()
    return (user, password) if user and password else None


def server_options() -> dict[str, Any]:
    """Dirección y puerto del servidor.

    Los hostings como Render indican el puerto en ``PORT``; entonces se
    escucha en todas las interfaces. En local se usan los valores de Gradio.
    """
    port = os.getenv("PORT")
    return {"server_name": "0.0.0.0", "server_port": int(port)} if port else {}


def reset() -> tuple[list[Message], None, str, str]:
    """Empieza una conversación nueva (agente y memoria nuevos)."""
    return [], None, EMPTY_PANEL, ""


with gr.Blocks(title=TITLE) as demo:
    gr.HTML(HEADER)
    session = gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=7, elem_classes="card"):
            chatbot = gr.Chatbot(
                elem_id="chat",
                show_label=False,
                height=540,
                buttons=["copy"],
                placeholder=(
                    "### ¿Qué quieres descubrir?\n"
                    "Elige una sugerencia o escribe tu pregunta."
                ),
            )
            with gr.Row(elem_id="composer"):
                textbox = gr.Textbox(
                    placeholder="Pregunta sobre Tenerife…",
                    show_label=False,
                    max_lines=4,
                    max_length=APP.max_message_chars,
                    scale=6,
                    container=False,
                )
                send = gr.Button("Enviar", variant="primary", scale=1, min_width=90)
            with gr.Row(elem_classes="chips"):
                chips = [
                    gr.Button(example, size="sm", elem_classes="chip")
                    for example in EXAMPLES
                ]
        with gr.Column(scale=4, elem_classes="card"):
            gr.HTML('<div class="panel-title">Observabilidad del turno</div>')
            panel = gr.HTML(EMPTY_PANEL)
            new_chat = gr.Button("↺ Nueva conversación", variant="secondary")
    gr.HTML(f'<div class="app-footer">{FOOTER}</div>')

    outputs = [chatbot, textbox, session, panel]
    textbox.submit(respond, [textbox, chatbot, session], outputs)
    send.click(respond, [textbox, chatbot, session], outputs)
    for chip in chips:
        # El valor de un botón es su texto: el ejemplo se envía con un clic.
        chip.click(respond, [chip, chatbot, session], outputs)
    new_chat.click(reset, None, [chatbot, session, panel, textbox])
    # La papelera del chat también borra la memoria del agente.
    chatbot.clear(reset, None, [chatbot, session, panel, textbox])

demo.queue(default_concurrency_limit=APP.concurrency)

if __name__ == "__main__":
    try:
        shared_tools()  # Índice listo antes de la primera pregunta.
    except Exception as err:
        print(f"Índice no disponible al arrancar: {err}")
    demo.launch(
        theme=THEME,
        css=CSS,
        auth=credentials(),
        auth_message="Demo privada del asistente turístico de Tenerife.",
        **server_options(),
    )
