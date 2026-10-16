"""Tests del agente, la memoria y el system prompt sin llamar a la API."""

from datetime import date

import pytest
from google.genai import types

from src.agent import EMPTY_ANSWER, TouristAgent
from src.config import AgentConfig, MemoryConfig
from src.llm import LLMError
from src.memory import ConversationMemory, Turn
from src.prompts import calendar, system_prompt

# --- Dobles de prueba -------------------------------------------------------


def text_chunk(text, prompt_tokens=100):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)])
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=10
        ),
    )


def call_chunk(*calls, prompt_tokens=100):
    parts = [
        types.Part(function_call=types.FunctionCall(id=f"c{n}", name=name, args=args))
        for n, (name, args) in enumerate(calls)
    ]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=5
        ),
    )


class FakeLLM:
    """LLM con respuestas programadas: cada elemento es una petición.

    Un elemento es una lista de fragmentos (streaming) o una excepción.
    """

    last_model = "fake-model"

    def __init__(self, *script, summary="Viaja con niños a Masca."):
        self.script = list(script)
        self.requests = []
        self.summary = summary

    def generate_stream(self, contents, system_instruction, tools, force_text):
        self.requests.append(
            {
                "contents": list(contents),
                "system": system_instruction,
                "force": force_text,
            }
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        yield from step

    def generate(self, prompt):
        self.requests.append({"summary_prompt": prompt})
        return text_chunk(self.summary)


class FakeTools:
    def __init__(self):
        self.executed = []

    def gemini_tools(self):
        return []

    def execute(self, name, args):
        self.executed.append((name, args))
        if name == "buscar_guia":
            return {
                "ok": True,
                "fragmentos": [
                    {"fuente": "TENERIFE.pdf, p. 22 · Masca", "texto": "..."}
                ],
            }
        if args.get("lugar") == "Mazca":
            return {"ok": False, "error": "lugar_desconocido", "mensaje": "..."}
        return {"ok": True, "cielo": "Despejado"}


def make_agent(*script, max_rounds=4, memory=None):
    llm, tools = FakeLLM(*script), FakeTools()
    agent = TouristAgent(
        client=llm,
        tools=tools,
        memory=memory or ConversationMemory(),
        config=AgentConfig(max_rounds=max_rounds),
    )
    return agent, llm, tools


# --- System prompt ----------------------------------------------------------


def test_calendar_maps_weekdays_to_iso_dates():
    lines = calendar(date(2026, 10, 16)).splitlines()
    assert lines[0] == "- sábado 17/10/2026 = 2026-10-17"
    assert lines[1] == "- domingo 18/10/2026 = 2026-10-18"
    assert lines[-1] == "- sábado 31/10/2026 = 2026-10-31"
    assert len(lines) == 15


def test_system_prompt_has_date_rules_and_summary():
    prompt = system_prompt(date(2026, 10, 16), summary="Viaja con niños.")
    assert "Hoy es viernes 16/10/2026 = 2026-10-16" in prompt
    assert "buscar_guia" in prompt and "solo puedes ayudar con Tenerife" in prompt
    assert "Viaja con niños." in prompt
    assert "Resumen" not in system_prompt(date(2026, 10, 16))


# --- Bucle del agente -------------------------------------------------------


def test_plain_answer_uses_no_tools():
    agent, llm, tools = make_agent(
        [text_chunk("¡Hola! "), text_chunk("¿Qué te apetece?")]
    )
    chunks = list(agent.chat_stream("hola"))

    assert chunks == ["¡Hola! ", "¿Qué te apetece?"]
    assert tools.executed == []
    assert agent.last_trace.rounds == 1
    assert agent.last_trace.first_token_s is not None
    assert len(agent.memory.turns) == 1


def test_parallel_tool_calls_then_answer():
    agent, llm, tools = make_agent(
        [
            call_chunk(
                ("get_weather", {"fecha": "2026-10-17", "lugar": "Masca"}),
                ("buscar_guia", {"consulta": "dónde comer en Masca"}),
            )
        ],
        [text_chunk("Sol en Masca (TENERIFE.pdf, p. 22).", prompt_tokens=900)],
    )
    reply = agent.chat("¿Tiempo el sábado en Masca y dónde comer?")

    assert reply.text == "Sol en Masca (TENERIFE.pdf, p. 22)."
    assert [name for name, _ in tools.executed] == ["get_weather", "buscar_guia"]
    assert reply.trace.rounds == 2
    assert reply.trace.sources == ["TENERIFE.pdf, p. 22 · Masca"]
    assert reply.trace.prompt_tokens == 1000
    assert reply.trace.last_prompt_tokens == 900

    # La segunda petición lleva la llamada del modelo y los dos resultados.
    second = llm.requests[1]["contents"]
    responses = second[-1].parts
    assert [p.function_response.name for p in responses] == [
        "get_weather",
        "buscar_guia",
    ]
    assert responses[0].function_response.id == "c0"


def test_failed_tool_lets_the_model_correct_itself():
    agent, llm, tools = make_agent(
        [call_chunk(("get_weather", {"fecha": "2026-10-17", "lugar": "Mazca"}))],
        [call_chunk(("get_weather", {"fecha": "2026-10-17", "lugar": "Masca"}))],
        [text_chunk("Despejado en Masca.")],
    )
    reply = agent.chat("¿Tiempo el sábado en Mazca?")

    assert [c.ok for c in reply.trace.tool_calls] == [False, True]
    assert reply.trace.tool_calls[0].error == "lugar_desconocido"
    assert reply.trace.rounds == 3


def test_round_limit_forces_a_text_answer():
    looping = [call_chunk(("buscar_guia", {"consulta": "playas"}))]
    agent, llm, tools = make_agent(
        *[looping] * 2, [text_chunk("Con lo que tengo: ...")], max_rounds=2
    )
    reply = agent.chat("playas")

    assert len(tools.executed) == 2
    assert [r["force"] for r in llm.requests] == [False, False, True]
    assert reply.trace.forced_final is True
    assert reply.text == "Con lo que tengo: ..."


def test_llm_error_is_shown_and_turn_not_saved():
    agent, llm, tools = make_agent(
        LLMError("Se ha alcanzado el límite de uso gratuito.")
    )
    reply = agent.chat("hola")

    assert "límite de uso" in reply.text
    assert reply.trace.error
    assert agent.memory.turns == []


def test_empty_answer_gets_a_fallback():
    agent, llm, tools = make_agent([text_chunk("")], [text_chunk("")])
    assert agent.chat("¿?").text == EMPTY_ANSWER


# --- Memoria ----------------------------------------------------------------


def test_history_keeps_tool_steps_only_for_last_turn():
    agent, llm, tools = make_agent(
        [call_chunk(("buscar_guia", {"consulta": "Masca"}))],
        [text_chunk("Masca es un caserío.")],
        [text_chunk("¡De nada!")],
        [text_chunk("Adiós.")],
    )
    agent.chat("¿Qué es Masca?")
    agent.chat("gracias")
    # Tercera petición del usuario: el turno con herramientas ya no es el último.
    agent.chat("adiós")
    history = llm.requests[-1]["contents"]

    assert not any(
        p.function_call or p.function_response for c in history for p in c.parts
    )
    texts = [c.parts[0].text for c in history]
    assert texts == [
        "¿Qué es Masca?",
        "Masca es un caserío.",
        "gracias",
        "¡De nada!",
        "adiós",
    ]


def test_previous_turn_tools_are_sent_in_full():
    agent, llm, tools = make_agent(
        [call_chunk(("buscar_guia", {"consulta": "Masca"}))],
        [text_chunk("Masca es un caserío.")],
        [text_chunk("Sí.")],
    )
    agent.chat("¿Qué es Masca?")
    agent.chat("¿y está bien?")
    history = llm.requests[-1]["contents"]
    assert any(p.function_response for c in history for p in c.parts)


def test_memory_is_summarized_when_over_budget():
    memory = ConversationMemory(
        MemoryConfig(max_history_tokens=500, keep_recent_turns=2)
    )
    agent, llm, tools = make_agent(
        [text_chunk("Uno.", prompt_tokens=200)],
        [text_chunk("Dos.", prompt_tokens=400)],
        [text_chunk("Tres.", prompt_tokens=600)],
        [text_chunk("Cuatro.", prompt_tokens=300)],
        memory=memory,
    )
    agent.chat("Viajo con niños")
    agent.chat("Vamos a Masca")
    assert not agent.last_trace.memory_compacted

    reply = agent.chat("¿Qué ropa llevo?")
    assert reply.trace.memory_compacted is True
    assert memory.summary == "Viaja con niños a Masca."
    assert len(memory.turns) == 2
    assert "Viajero: Viajo con niños" in llm.requests[-1]["summary_prompt"]

    agent.chat("¿Y el Teide?")
    assert "Viaja con niños a Masca." in llm.requests[-1]["system"]


def test_failed_summary_keeps_full_history():
    def broken(prompt):
        raise LLMError("caído")

    memory = ConversationMemory(
        MemoryConfig(max_history_tokens=10, keep_recent_turns=1)
    )
    for text in ("uno", "dos"):
        user = types.Content(role="user", parts=[types.Part(text=text)])
        answer = types.Content(role="model", parts=[types.Part(text="ok")])
        memory.add(Turn(user, answer), prompt_tokens=50)

    assert memory.needs_compaction()
    assert memory.compact(broken) is False
    assert len(memory.turns) == 2


@pytest.mark.parametrize("turns", [0, 1])
def test_reset_forgets_everything(turns):
    agent, llm, tools = make_agent(*[[text_chunk("ok")]] * turns)
    for _ in range(turns):
        agent.chat("hola")
    agent.memory.summary = "algo"
    agent.reset()
    assert agent.memory.turns == [] and agent.memory.summary == ""


def test_empty_response_is_retried_once():
    agent, llm, tools = make_agent([text_chunk("")], [text_chunk("Ahora sí.")])
    reply = agent.chat("¿Qué tiempo hará el domingo en Tenerife?")
    assert reply.text == "Ahora sí."
    assert reply.trace.rounds == 2
