"""Agente conversacional: el bucle que une LLM, herramientas y memoria.

En cada turno:

1. Se envía al LLM el system prompt (con la fecha y el resumen de la
   conversación), el historial, el mensaje del usuario y las declaraciones de
   las herramientas.
2. Si el LLM pide herramientas (una o varias a la vez), nuestro código las
   ejecuta y le devuelve los resultados; es una "vuelta" del bucle.
3. Se repite hasta que el LLM responde con texto o se alcanza el límite de
   vueltas; en ese caso se le obliga a responder sin herramientas.
4. El turno se guarda en la memoria, que se compacta si supera su
   presupuesto de tokens.

La respuesta se emite en streaming y cada turno deja una traza (herramientas,
fuentes, tokens, latencias) para el notebook, la evaluación y la app.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from src.config import AGENT, AgentConfig
from src.llm import GeminiClient, LLMError
from src.memory import ConversationMemory, Turn, content_text
from src.prompts import system_prompt
from src.tools import ToolRegistry

logger = logging.getLogger(__name__)

EMPTY_ANSWER = "No he podido generar una respuesta. ¿Puedes reformular la pregunta?"


@dataclass
class ToolCallTrace:
    """Una llamada a herramienta dentro de un turno.

    ``result`` es lo que recibió el LLM (fragmentos, datos del tiempo...): lo
    usa el juez de la evaluación para comprobar la fidelidad de la respuesta.
    """

    name: str
    args: dict[str, Any]
    ok: bool
    error: str | None
    latency_ms: int
    sources: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    """Lo que ha ocurrido en un turno (observabilidad).

    Attributes:
        rounds: Peticiones al LLM en el bucle.
        tool_calls: Herramientas ejecutadas, en orden.
        prompt_tokens: Tokens de entrada sumados de todas las peticiones.
        output_tokens: Tokens generados sumados.
        thinking_tokens: Tokens de razonamiento sumados.
        last_prompt_tokens: Tokens del prompt de la última petición (mide el
            tamaño del contexto para la memoria).
        first_token_s: Segundos hasta el primer fragmento de texto.
        total_s: Duración total del turno.
        model: Modelo que respondió.
        forced_final: True si se alcanzó el límite de vueltas.
        finish_reasons: Motivo de fin de cada petición (``STOP``,
            ``MALFORMED_FUNCTION_CALL``...), para diagnosticar fallos.
        memory_compacted: True si se resumió el historial tras el turno.
        error: Mensaje de error si el turno falló.
    """

    rounds: int = 0
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    last_prompt_tokens: int = 0
    first_token_s: float | None = None
    total_s: float = 0.0
    model: str | None = None
    forced_final: bool = False
    finish_reasons: list[str | None] = field(default_factory=list)
    memory_compacted: bool = False
    error: str | None = None

    @property
    def sources(self) -> list[str]:
        """Fuentes de la guía consultadas en el turno, sin repetir."""
        seen: dict[str, None] = {}
        for call in self.tool_calls:
            seen.update(dict.fromkeys(call.sources))
        return list(seen)

    def add_usage(self, usage: types.GenerateContentResponseUsageMetadata | None):
        """Suma el consumo de tokens de una petición."""
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_token_count or 0
        self.output_tokens += usage.candidates_token_count or 0
        self.thinking_tokens += usage.thoughts_token_count or 0
        self.last_prompt_tokens = usage.prompt_token_count or 0


@dataclass
class AgentReply:
    """Respuesta completa de un turno."""

    text: str
    trace: TurnTrace


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    """Partes de un fragmento de respuesta (vacío si no trae contenido)."""
    if not chunk.candidates or not chunk.candidates[0].content:
        return []
    return list(chunk.candidates[0].content.parts or [])


def _finish_reason(chunk: types.GenerateContentResponse) -> str | None:
    """Motivo de fin de la respuesta, si el fragmento lo trae."""
    if not chunk.candidates or not chunk.candidates[0].finish_reason:
        return None
    reason = chunk.candidates[0].finish_reason
    return getattr(reason, "value", str(reason))


class TouristAgent:
    """Asistente turístico conversacional con herramientas y memoria.

    Attributes:
        last_trace: Traza del último turno.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        config: AgentConfig = AGENT,
    ):
        """Crea el agente.

        Args:
            client: Cliente del LLM. Por defecto, Gemini con ``src.config``.
            tools: Herramientas disponibles. Por defecto, las tres del
                asistente sobre el índice de la guía.
            memory: Memoria de la conversación. Por defecto, una vacía.
            config: Límite de vueltas del bucle.
        """
        self.client = client or GeminiClient()
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self.config = config
        self.last_trace = TurnTrace()

    def chat(self, message: str) -> AgentReply:
        """Responde a un mensaje y devuelve el texto completo con su traza."""
        text = "".join(self.chat_stream(message))
        return AgentReply(text=text, trace=self.last_trace)

    def chat_stream(self, message: str) -> Iterator[str]:
        """Responde a un mensaje en streaming.

        Args:
            message: Mensaje del usuario.

        Yields:
            Fragmentos de texto de la respuesta a medida que se generan. Al
            terminar, la traza del turno queda en :attr:`last_trace`.
        """
        trace = self.last_trace = TurnTrace()
        start = time.perf_counter()
        user = types.Content(role="user", parts=[types.Part(text=message)])
        steps: list[types.Content] = []
        system = system_prompt(summary=self.memory.summary)
        tools = self.tools.gemini_tools()

        retried_empty = False
        try:
            # Una vuelta más que el límite: la de la respuesta forzada.
            for round_number in range(1, self.config.max_rounds + 2):
                force_text = round_number > self.config.max_rounds
                trace.forced_final = force_text
                parts: list[types.Part] = []
                usage = finish = None
                stream = self.client.generate_stream(
                    self.memory.contents() + [user, *steps],
                    system_instruction=system,
                    tools=tools,
                    force_text=force_text,
                )
                for chunk in stream:
                    usage = chunk.usage_metadata or usage
                    finish = _finish_reason(chunk) or finish
                    for part in _parts(chunk):
                        parts.append(part)
                        if part.text and not part.thought:
                            if trace.first_token_s is None:
                                trace.first_token_s = time.perf_counter() - start
                            yield part.text
                trace.rounds = round_number
                trace.model = self.client.last_model
                trace.finish_reasons.append(finish)
                trace.add_usage(usage)

                model_content = types.Content(role="model", parts=parts)
                calls = [part.function_call for part in parts if part.function_call]
                empty = not calls and not content_text(model_content)
                if empty and not retried_empty and not force_text:
                    # Respuesta vacía (p. ej. una llamada mal formada que la API
                    # descarta): se repite la vuelta una vez.
                    logger.warning("Respuesta vacía (%s); se reintenta", finish)
                    retried_empty = True
                    continue
                if not calls or force_text:
                    break
                steps.append(model_content)
                steps.append(
                    types.Content(
                        role="user", parts=[self._run_tool(c, trace) for c in calls]
                    )
                )
        except LLMError as err:
            trace.error = str(err)
            trace.total_s = time.perf_counter() - start
            yield f"\n\n⚠️ {err}"
            return

        if not content_text(model_content):
            yield EMPTY_ANSWER
            model_content = types.Content(
                role="model", parts=[types.Part(text=EMPTY_ANSWER)]
            )
        self.memory.add(Turn(user, model_content, steps), trace.last_prompt_tokens)
        if self.memory.needs_compaction():
            trace.memory_compacted = self.memory.compact(self._summarize)
        trace.total_s = time.perf_counter() - start

    def _run_tool(self, call: types.FunctionCall, trace: TurnTrace) -> types.Part:
        """Ejecuta una herramienta y prepara su resultado para el LLM."""
        args = dict(call.args or {})
        start = time.perf_counter()
        result = self.tools.execute(call.name, args)
        trace.tool_calls.append(
            ToolCallTrace(
                name=call.name,
                args=args,
                ok=bool(result.get("ok")),
                error=result.get("error"),
                latency_ms=round((time.perf_counter() - start) * 1000),
                sources=[f["fuente"] for f in result.get("fragmentos", [])],
                result=result,
            )
        )
        return types.Part(
            function_response=types.FunctionResponse(
                id=call.id, name=call.name, response=result
            )
        )

    def _summarize(self, prompt: str) -> str:
        """Pide al LLM el resumen de los turnos antiguos (sin herramientas)."""
        response = self.client.generate(prompt)
        self.last_trace.add_usage(response.usage_metadata)
        return response.text or ""

    def reset(self) -> None:
        """Empieza una conversación nueva."""
        self.memory.reset()
        self.last_trace = TurnTrace()
