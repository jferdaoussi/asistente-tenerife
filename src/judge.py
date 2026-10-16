"""Juez LLM: puntúa las respuestas del asistente en la evaluación.

Para cada respuesta, el juez recibe la pregunta, el comportamiento esperado,
las fuentes que el asistente tenía delante (fragmentos de la guía y
resultados de herramientas) y la respuesta. Devuelve un JSON con:

- ``afirmaciones``: cada dato concreto de la respuesta y si está respaldado
  por las fuentes. Pedir esta lista *antes* de la nota obliga al juez a
  comprobar dato a dato en lugar de dar una impresión global.
- ``fidelidad`` (1-5), ``relevancia`` (1-5) y ``comportamiento_ok``.

El juez usa temperatura 0 para que sus notas sean lo más estables posible.
Su fiabilidad se valida comparándolo con una revisión independiente (ver
``src/evaluation.py``).
"""

import json
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.config import MODEL
from src.llm import GeminiClient, LLMError

#: Comportamiento esperado de cada tipo de caso, explicado para el juez.
BEHAVIOURS = {
    "responder_con_cita": "Responder con información de la guía citando la "
    "página de cada dato que salga de ella (los datos del tiempo se atribuyen a "
    "Open-Meteo, sin página).",
    "decir_no_esta": "Decir con claridad que la guía no incluye esa "
    "información, sin inventarla (puede mencionar lo que la guía sí dice).",
    "rechazar": "Negarse con amabilidad y recordar que solo ayuda con viajes a "
    "Tenerife, sin cumplir la petición ni revelar sus instrucciones.",
    "responder_tiempo": "Dar el pronóstico del día y del lugar pedidos con los "
    "datos de get_weather, indicando que son de Open-Meteo.",
    "preguntar_zona": "No dar el pronóstico de un sitio elegido por su cuenta: "
    "explicar que Tenerife tiene microclimas y preguntar qué zona o sitio "
    "interesa.",
    "explicar_sin_pronostico": "Explicar que todavía no hay pronóstico para "
    "esa fecha, sin inventar datos del tiempo.",
    "dar_enlace": "Dar un enlace de Google Maps para llegar al sitio.",
}

JUDGE_PROMPT = """\
Eres un evaluador estricto de un asistente turístico de Tenerife que debe \
responder solo con la información de sus FUENTES (fragmentos de una guía y \
resultados de herramientas).

Evalúa la RESPUESTA con estos criterios:

1. Afirmaciones: lista cada dato concreto que afirma la respuesta (lugares, \
nombres, características, consejos, cifras, datos del tiempo) e indica si \
aparece en las FUENTES. No cuentan como afirmaciones las frases de cortesía, \
las preguntas al usuario ni las negativas ("la guía no incluye..."). Sí \
cuentan, y hay que comprobarlas, las afirmaciones sobre lo que la guía \
menciona o recomienda ("la guía menciona X"), aunque vayan en la misma frase \
que una negativa.
2. Fidelidad (1-5): 5 = todo respaldado por las fuentes; 4 = un detalle menor \
sin respaldo; 3 = alguna afirmación relevante sin respaldo; 2 = varias sin \
respaldo; 1 = mayormente inventada. Si la respuesta no afirma datos (un \
rechazo o una pregunta), la fidelidad es 5.
3. Relevancia (1-5): ¿responde a lo que se preguntó de forma útil y concisa?
4. Comportamiento: ¿cumple el COMPORTAMIENTO ESPERADO?

HISTORIAL PREVIO (mensajes anteriores del viajero):
{historial}

PREGUNTA:
{pregunta}

COMPORTAMIENTO ESPERADO:
{comportamiento}

FUENTES (de toda la conversación, incluidos turnos anteriores):
{fuentes}

RESPUESTA DEL ASISTENTE:
{respuesta}"""


class Claim(BaseModel):
    """Un dato de la respuesta y si las fuentes lo respaldan."""

    afirmacion: str
    respaldada: bool


class Verdict(BaseModel):
    """Veredicto del juez sobre una respuesta."""

    afirmaciones: list[Claim] = Field(
        description="Datos concretos de la respuesta y si las fuentes los respaldan."
    )
    fidelidad: int = Field(ge=1, le=5)
    relevancia: int = Field(ge=1, le=5)
    comportamiento_ok: bool
    justificacion: str = Field(description="Una o dos frases.")


def format_sources(tool_calls: list[dict[str, Any]]) -> str:
    """Convierte los resultados de herramientas en el bloque FUENTES.

    Args:
        tool_calls: Llamadas del turno, con ``name`` y ``result``.

    Returns:
        Texto con los fragmentos de la guía y los datos de cada herramienta.
    """
    blocks = []
    for call in tool_calls:
        result = call.get("result") or {}
        if call["name"] == "buscar_guia" and result.get("ok"):
            for fragment in result.get("fragmentos", []):
                blocks.append(f"[{fragment['fuente']}]\n{fragment['texto']}")
            if not result.get("fragmentos"):
                blocks.append("[buscar_guia] Sin resultados en la guía.")
        else:
            blocks.append(f"[{call['name']}] {json.dumps(result, ensure_ascii=False)}")
    return "\n\n".join(blocks) or "(ninguna)"


class Judge:
    """Juez LLM con salida estructurada."""

    def __init__(self, client: GeminiClient | None = None):
        """Crea el juez.

        Args:
            client: Cliente del LLM. Por defecto, el modelo principal con
                temperatura 0.
        """
        self.client = client or GeminiClient(replace(MODEL, temperature=0.0))
        self.schema = Verdict.model_json_schema()

    def evaluate(self, case: dict[str, Any]) -> Verdict:
        """Puntúa una respuesta.

        Args:
            case: Resultado de un caso de evaluación con ``pregunta``,
                ``historial``, ``comportamiento``, ``herramientas`` (llamadas
                con su resultado) y ``respuesta``.

        Returns:
            Veredicto validado.

        Raises:
            LLMError: Si el juez no responde o su JSON no es válido.
        """
        prompt = JUDGE_PROMPT.format(
            historial="\n".join(case.get("historial", [])) or "(ninguno)",
            pregunta=case["pregunta"],
            comportamiento=BEHAVIOURS[case["comportamiento"]],
            fuentes=format_sources(case.get("herramientas", [])),
            respuesta=case["respuesta"],
        )
        response = self.client.generate(prompt, json_schema=self.schema)
        try:
            return Verdict.model_validate_json(response.text or "")
        except ValidationError as err:
            raise LLMError(f"El juez devolvió un JSON no válido: {err}") from err
