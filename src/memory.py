"""Memoria de la conversación con presupuesto de tokens.

La API de Gemini no recuerda nada entre peticiones: el historial se reenvía
entero en cada turno. Esta clase decide qué se reenvía (ver
:class:`src.config.MemoryConfig`):

- El último turno va completo, con las llamadas a herramientas y sus
  resultados.
- Los turnos anteriores van "adelgazados": solo la pregunta y la respuesta.
- Cuando el prompt supera el presupuesto, los turnos más antiguos se resumen
  con el LLM en un párrafo, que pasa al system prompt.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from google.genai import types

from src.config import MEMORY, MemoryConfig
from src.llm import LLMError

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
Resume esta conversación entre un viajero y un asistente turístico de \
Tenerife en un máximo de {max_words} palabras, en español. Conserva solo lo \
útil para seguir ayudando: destino o zona de la isla y alojamiento, fechas, \
con quién viaja, intereses, preferencias y restricciones, y los sitios ya \
recomendados (con su página de la guía). Omite saludos y detalles del tiempo \
ya pasados. Si hay un resumen previo, intégralo.

Resumen previo:
{previous}

Conversación:
{transcript}"""


def content_text(content: types.Content) -> str:
    """Texto visible de un mensaje (sin razonamiento ni llamadas)."""
    return "".join(
        part.text for part in content.parts or [] if part.text and not part.thought
    )


@dataclass
class Turn:
    """Un intercambio completo: pregunta, pasos intermedios y respuesta.

    Attributes:
        user: Mensaje del usuario.
        answer: Respuesta final del modelo.
        steps: Llamadas a herramientas del modelo y sus resultados, en orden.
    """

    user: types.Content
    answer: types.Content
    steps: list[types.Content] = field(default_factory=list)

    def contents(self, full: bool) -> list[types.Content]:
        """Mensajes del turno, con o sin los pasos de herramientas."""
        if full:
            return [self.user, *self.steps, self.answer]
        return [self.user, self.answer]

    def transcript(self) -> str:
        """El turno en texto plano, para resumirlo."""
        return (
            f"Viajero: {content_text(self.user)}\n"
            f"Asistente: {content_text(self.answer)}"
        )


class ConversationMemory:
    """Historial de la conversación con presupuesto de tokens.

    Attributes:
        turns: Turnos conservados, del más antiguo al más reciente.
        summary: Resumen de los turnos ya eliminados.
        last_prompt_tokens: Tokens del prompt de la última petición del
            último turno, según la API.
    """

    def __init__(self, config: MemoryConfig = MEMORY):
        """Crea una memoria vacía.

        Args:
            config: Presupuesto y número de turnos recientes.
        """
        self.config = config
        self.turns: list[Turn] = []
        self.summary = ""
        self.last_prompt_tokens = 0

    def contents(self) -> list[types.Content]:
        """Historial que se envía al LLM en la siguiente petición."""
        history: list[types.Content] = []
        for position, turn in enumerate(self.turns, start=1):
            history += turn.contents(full=position == len(self.turns))
        return history

    def add(self, turn: Turn, prompt_tokens: int) -> None:
        """Guarda un turno terminado.

        Args:
            turn: Turno completo.
            prompt_tokens: Tokens del prompt de su última petición.
        """
        self.turns.append(turn)
        self.last_prompt_tokens = prompt_tokens

    def needs_compaction(self) -> bool:
        """True si se ha superado el presupuesto y hay turnos que resumir."""
        return (
            self.last_prompt_tokens > self.config.max_history_tokens
            and len(self.turns) > self.config.keep_recent_turns
        )

    def compact(self, summarize: Callable[[str], str]) -> bool:
        """Resume los turnos antiguos y conserva solo los recientes.

        Args:
            summarize: Función que recibe un prompt y devuelve el resumen
                (normalmente, una llamada al LLM).

        Returns:
            True si se ha compactado; False si el resumen ha fallado, en cuyo
            caso se conserva todo el historial y se reintentará en el
            siguiente turno.
        """
        keep = self.config.keep_recent_turns
        old, recent = self.turns[:-keep], self.turns[-keep:]
        prompt = SUMMARY_PROMPT.format(
            max_words=self.config.summary_max_words,
            previous=self.summary or "(ninguno)",
            transcript="\n\n".join(turn.transcript() for turn in old),
        )
        try:
            summary = summarize(prompt).strip()
        except LLMError as err:
            logger.warning("No se pudo resumir el historial: %s", err)
            return False
        if not summary:
            return False
        self.summary = summary
        self.turns = recent
        logger.info("Historial compactado: %d turnos resumidos", len(old))
        return True

    def reset(self) -> None:
        """Olvida toda la conversación."""
        self.turns.clear()
        self.summary = ""
        self.last_prompt_tokens = 0
