"""Tests de la evaluación y del juez sin llamar a la API."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src import evaluation
from src.evaluation import (
    ThrottledClient,
    accepted_dates,
    check_case,
    cited_pages,
    contains_evidence,
    judge_agreement,
    load_questions,
    retrieval_summary,
)
from src.judge import BEHAVIOURS, Judge, format_sources
from src.llm import LLMError

FRIDAY = date(2026, 10, 16)

# --- Conjunto de evaluación -------------------------------------------------


def test_question_set_is_consistent():
    questions = load_questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) == 36
    for q in questions:
        assert q["comportamiento"] in BEHAVIOURS
        assert bool(q["evidencia"]) == bool(q["paginas"])
        if spec := q.get("tiempo"):
            accepted_dates(spec["fecha"], FRIDAY)  # no lanza


# --- Utilidades -------------------------------------------------------------


def test_cited_pages_expands_ranges():
    text = "Masca (TENERIFE.pdf, p. 22) y La Orotava (TENERIFE.pdf, p. 8-11)."
    assert cited_pages(text) == {22, 8, 9, 10, 11}
    assert cited_pages("sin citas") == set()


def test_evidence_matching_ignores_case_and_spacing():
    assert contains_evidence("el  Molino de\nChano vende", ["molino de chano"])
    assert not contains_evidence("otro texto", ["molino de chano"])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("+1", {"2026-10-17"}),
        ("sábado", {"2026-10-17"}),
        ("domingo", {"2026-10-18"}),
        ("viernes", {"2026-10-23", "2026-10-16"}),  # hoy es viernes: ambiguo
    ],
)
def test_accepted_dates(spec, expected):
    assert accepted_dates(spec, FRIDAY) == expected


# --- Comprobaciones automáticas ---------------------------------------------


def call(name, args, ok=True, fragments=()):
    result = {"ok": ok}
    if name == "buscar_guia":
        result["fragmentos"] = [{"fuente": "f", "texto": t} for t in fragments]
    return {"name": name, "args": args, "ok": ok, "result": result}


def test_check_guide_answer():
    question = {
        "herramientas": ["buscar_guia"],
        "comportamiento": "responder_con_cita",
        "evidencia": ["molino de Chano"],
        "paginas": [9],
    }
    record = {
        "respuesta": "En el molino de Chano (TENERIFE.pdf, p. 8-9).",
        "herramientas": [call("buscar_guia", {}, fragments=["el molino de Chano"])],
    }
    checks = check_case(question, record, FRIDAY)
    assert checks == {
        "herramientas_ok": True,
        "tiempo_ok": None,
        "recuperado_ok": True,
        "cita_ok": True,
        "enlace_ok": None,
    }


def test_check_weather_place_and_date():
    question = {
        "herramientas": ["get_weather"],
        "comportamiento": "responder_tiempo",
        "evidencia": [],
        "paginas": [],
        "tiempo": {"lugar": ["La Orotava"], "fecha": "domingo"},
    }
    good = {
        "respuesta": "",
        "herramientas": [
            call("get_weather", {"lugar": "La Orotava", "fecha": "2026-10-18"})
        ],
    }
    wrong_day = {
        "respuesta": "",
        "herramientas": [
            call("get_weather", {"lugar": "La Orotava", "fecha": "2026-10-17"})
        ],
    }
    assert check_case(question, good, FRIDAY)["tiempo_ok"] is True
    assert check_case(question, wrong_day, FRIDAY)["tiempo_ok"] is False


def test_check_no_forecast_when_zone_is_missing():
    question = {
        "herramientas": [],
        "comportamiento": "preguntar_zona",
        "evidencia": [],
        "paginas": [],
    }
    asked = {"respuesta": "¿Qué zona?", "herramientas": []}
    guessed = {
        "respuesta": "Sol en Santa Cruz.",
        "herramientas": [call("get_weather", {"lugar": "Santa Cruz de Tenerife"})],
    }
    assert check_case(question, asked, FRIDAY)["herramientas_ok"] is True
    assert check_case(question, guessed, FRIDAY)["herramientas_ok"] is False


# --- Recuperación -----------------------------------------------------------


def test_retrieval_summary_metrics():
    df = pd.DataFrame(
        {
            "chunker": ["a"] * 4,
            "id": ["q1", "q2", "q3", "q4"],
            "n_fragmentos": [10] * 4,
            "rank": [1, 2, 0, 4],
            "sim_top": [0.8] * 4,
            "sim_acierto": [0.8, 0.7, None, 0.6],
        }
    )
    row = retrieval_summary(df).loc["a"]
    assert row["recall@1"] == 0.25
    assert row["recall@4"] == 0.75
    assert row["MRR"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_evaluate_retrieval_with_fake_embedder(tmp_path):
    from tests.fakes import FakeEmbedder

    questions = [
        {
            "id": "q1",
            "pregunta": "subir al teide",
            "evidencia": ["teleféricos del Teide"],
        },
        {"id": "q2", "pregunta": "sin evidencia", "evidencia": [], "historial": []},
    ]
    df = evaluation.evaluate_retrieval(
        questions, embedder=FakeEmbedder(), persist_dir=tmp_path
    )
    assert set(df["chunker"]) == {"por_sitio", "tamano_fijo", "langchain"}
    assert set(df["id"]) == {"q1"}
    assert (df["rank"] >= 1).all()  # la evidencia aparece en los 4 primeros


# --- Ritmo de peticiones ----------------------------------------------------


def test_throttled_client_waits_when_limit_is_reached(monkeypatch):
    clock = {"t": 0.0}
    sleeps = []
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(evaluation.time, "sleep", sleeps.append)
    inner = SimpleNamespace(last_model="m", generate=lambda *a, **k: "ok")
    client = ThrottledClient(inner, rpm=2)

    client.generate("a")
    client.generate("b")
    assert sleeps == []
    clock["t"] = 10.0
    client.generate("c")
    assert sleeps == [pytest.approx(50.5)]


# --- Juez ---------------------------------------------------------------------


def test_format_sources_includes_fragments_and_tool_data():
    text = format_sources(
        [
            call("buscar_guia", {}, fragments=["Masca es un caserío"]),
            {"name": "get_weather", "args": {}, "ok": True, "result": {"cielo": "Sol"}},
        ]
    )
    assert "Masca es un caserío" in text
    assert '"cielo": "Sol"' in text
    assert format_sources([]) == "(ninguna)"


class FakeJudgeLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate(self, prompt, json_schema=None):
        self.prompts.append((prompt, json_schema))
        return SimpleNamespace(text=self.text)


CASE = {
    "pregunta": "¿Dónde comprar gofio?",
    "historial": [],
    "comportamiento": "responder_con_cita",
    "herramientas": [call("buscar_guia", {}, fragments=["molino de Chano"])],
    "respuesta": "En el molino de Chano.",
}


def test_judge_parses_structured_verdict():
    llm = FakeJudgeLLM(
        '{"afirmaciones": [{"afirmacion": "molino de Chano", "respaldada": true}],'
        ' "fidelidad": 5, "relevancia": 4, "comportamiento_ok": true,'
        ' "justificacion": "ok"}'
    )
    verdict = Judge(client=llm).evaluate(CASE)
    assert verdict.fidelidad == 5
    prompt, schema = llm.prompts[0]
    assert "molino de Chano" in prompt and "¿Dónde comprar gofio?" in prompt
    assert schema["properties"]["fidelidad"]["maximum"] == 5


def test_judge_rejects_invalid_json():
    with pytest.raises(LLMError):
        Judge(client=FakeJudgeLLM('{"fidelidad": 9}')).evaluate(CASE)


# --- Validación del juez ------------------------------------------------------


def verdict(fid, rel=4, ok=True):
    return {"fidelidad": fid, "relevancia": rel, "comportamiento_ok": ok}


def test_judge_agreement_reliable():
    records = [{"id": f"q{i}", "juez": verdict(5)} for i in range(5)]
    reference = [{"id": f"q{i}", **verdict(4 if i else 5)} for i in range(5)]
    result = judge_agreement(records, reference)
    assert result["fidelidad_±1"] == 1.0
    assert result["fiable"] is True


def test_judge_agreement_flags_missed_hallucination():
    records = [{"id": "q1", "juez": verdict(4)}, {"id": "q2", "juez": verdict(5)}]
    reference = [{"id": "q1", **verdict(3)}, {"id": "q2", **verdict(5)}]
    result = judge_agreement(records, reference)
    assert result["fidelidad_±1"] == 1.0
    assert result["alucinaciones_no_detectadas"] == ["q1"]
    assert result["fiable"] is False
