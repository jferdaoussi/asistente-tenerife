"""Evaluación del asistente con un conjunto fijo de preguntas.

El conjunto (``eval/preguntas.jsonl``) incluye preguntas que responde la
guía, casos que no cubre, preguntas fuera de dominio, del tiempo, combinadas,
de seguimiento e intentos de manipulación. Cada pregunta de la guía lleva una
**frase de evidencia** literal ("molino de Chano"): un fragmento recuperado
es correcto si la contiene, sea cual sea el chunker que lo generó.

Pasos:

1. **Recuperación** (:func:`evaluate_retrieval`): recall@k y MRR de los tres
   chunkers con las mismas preguntas. Solo usa embeddings.
2. **Agente** (:func:`run_agent_eval`): el asistente completo responde cada
   pregunta y se comprueban de forma automática las herramientas usadas, los
   argumentos del tiempo, las citas de página y si recuperó la evidencia.
3. **Juez** (:func:`judge_records`): un LLM puntúa fidelidad, relevancia y
   comportamiento.
4. **Validación del juez** (:func:`judge_agreement`): se compara con una
   revisión independiente hecha por un modelo más potente.

Los resultados se guardan en ``eval/resultados/`` y se reutilizan al volver
a ejecutar, para no depender de la cuota de la API.

Uso: ``python -m src.evaluation --paso todo --thinking MINIMAL``.
"""

import argparse
import json
import logging
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import TouristAgent
from src.chunking import chunk_by_section, chunk_fixed, chunk_recursive
from src.config import CHROMA_DIR, MODEL, ROOT_DIR
from src.embeddings import GeminiEmbedder
from src.judge import Judge
from src.llm import GeminiClient, LLMError
from src.loader import load_pages
from src.memory import ConversationMemory
from src.prompts import WEEKDAYS
from src.rag import GuideIndex
from src.tools import ToolRegistry
from src.weather import today as canary_today

logger = logging.getLogger(__name__)

EVAL_DIR = ROOT_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "preguntas.jsonl"
RESULTS_DIR = EVAL_DIR / "resultados"
REVIEW_PATH = EVAL_DIR / "revision_claude.jsonl"
QUERY_CACHE_PATH = EVAL_DIR / "cache" / "embeddings_preguntas.json"

#: Estrategias de chunking comparadas (ver docs/decisiones.md, D4).
CHUNKERS: dict[str, Callable] = {
    "por_sitio": chunk_by_section,
    "tamano_fijo": chunk_fixed,
    "langchain": chunk_recursive,
}

#: Comportamientos en los que no debe darse un pronóstico.
NO_FORECAST = {"rechazar", "preguntar_zona", "explicar_sin_pronostico"}

PAGE_CITATION = re.compile(r"p\.\s*(\d+)(?:\s*[-–]\s*(\d+))?")


# --- Utilidades ---------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Lee el conjunto de evaluación (una pregunta JSON por línea)."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize(text: str) -> str:
    """Minúsculas y espacios simples, para buscar la evidencia en un texto."""
    return re.sub(r"\s+", " ", text).lower()


def contains_evidence(text: str, evidence: list[str]) -> bool:
    """True si el texto contiene alguna de las frases de evidencia."""
    text = normalize(text)
    return any(normalize(phrase) in text for phrase in evidence)


def cited_pages(text: str) -> set[int]:
    """Páginas citadas en una respuesta ("p. 8" o "p. 8-11")."""
    pages: set[int] = set()
    for start, end in PAGE_CITATION.findall(text):
        first = int(start)
        pages.update(range(first, int(end or first) + 1))
    return pages


def accepted_dates(spec: str, today: date) -> set[str]:
    """Fechas válidas para una fecha relativa del conjunto de evaluación.

    Args:
        spec: "+N" (dentro de N días) o un día de la semana ("sábado").
        today: Fecha de ejecución.

    Returns:
        Fechas ISO aceptadas. Para un día de la semana, el próximo; si hoy es
        ese mismo día, también hoy (la expresión es ambigua).
    """
    if spec.startswith("+"):
        return {(today + timedelta(days=int(spec[1:]))).isoformat()}
    target = WEEKDAYS.index(spec)
    ahead = (target - today.weekday()) % 7 or 7
    dates = {(today + timedelta(days=ahead)).isoformat()}
    if today.weekday() == target:
        dates.add(today.isoformat())
    return dates


# --- 1. Recuperación ----------------------------------------------------------


class CachedEmbedder:
    """Embedder que recuerda los vectores de las preguntas, también en disco.

    Las mismas preguntas se buscan en tres índices y en cada ejecución de la
    evaluación; así cada una se convierte en vector una sola vez (la capa
    gratuita admite 100 peticiones de embeddings por minuto).
    """

    def __init__(self, embedder: Any = None, path: Path | None = None):
        """Envuelve un embedder.

        Args:
            embedder: Embedder real (por defecto, el de Gemini).
            path: Archivo JSON de caché; ``None`` para no usar disco.
        """
        self.embedder = embedder or GeminiEmbedder()
        self.path = path
        self._queries: dict[str, list[float]] = {}
        if path and path.exists():
            self._queries = json.loads(path.read_text(encoding="utf-8"))

    def embed_documents(self, texts: list[str], titles: list[str]) -> list:
        """Delega en el embedder envuelto."""
        return self.embedder.embed_documents(texts, titles)

    def embed_query(self, query: str) -> list[float]:
        """Vector de una pregunta, calculado solo la primera vez."""
        if query not in self._queries:
            self._queries[query] = self.embedder.embed_query(query)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._queries), encoding="utf-8")
        return self._queries[query]


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    k: int = 4,
    embedder: Any = None,
    persist_dir: Path = CHROMA_DIR,
) -> pd.DataFrame:
    """Mide la recuperación de cada chunker con las mismas preguntas.

    Se busca sin umbral de similitud, para medir el orden de los resultados.
    Se excluyen las preguntas de seguimiento ("¿y dónde comer allí?"): sin la
    conversación previa no tienen sentido como consulta, y en el asistente
    real es el LLM quien las reescribe antes de buscar.

    Args:
        questions: Conjunto de evaluación; se usan las que tienen evidencia y
            no dependen de turnos anteriores.
        k: Fragmentos recuperados por pregunta.
        embedder: Embedder a usar (por defecto, Gemini con caché).
        persist_dir: Carpeta de los índices de Chroma.

    Returns:
        Una fila por chunker y pregunta: posición del primer acierto
        (``rank``, 0 si no aparece en los k primeros) y similitudes.
    """
    embedder = embedder or CachedEmbedder(path=QUERY_CACHE_PATH)
    pages = load_pages()
    with_evidence = [q for q in questions if q["evidencia"] and not q.get("historial")]
    rows = []
    for name, chunker in CHUNKERS.items():
        chunks = chunker(pages)
        index = GuideIndex(
            embedder=embedder, persist_dir=persist_dir, collection_name=f"eval_{name}"
        )
        index.build(chunks)
        for question in with_evidence:
            results = index.search(question["pregunta"], k=k, min_similarity=0)
            rank = next(
                (
                    position
                    for position, result in enumerate(results, start=1)
                    if contains_evidence(result.text, question["evidencia"])
                ),
                0,
            )
            rows.append(
                {
                    "chunker": name,
                    "id": question["id"],
                    "n_fragmentos": len(chunks),
                    "rank": rank,
                    "sim_top": results[0].similarity if results else None,
                    "sim_acierto": results[rank - 1].similarity if rank else None,
                }
            )
    return pd.DataFrame(rows)


def retrieval_summary(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Recall@1, recall@k y MRR por chunker.

    Args:
        df: Resultado de :func:`evaluate_retrieval`.
        k: Número de fragmentos recuperados.

    Returns:
        Una fila por chunker.
    """
    hit = df["rank"].between(1, k)
    summary = df.assign(
        r1=df["rank"] == 1,
        rk=hit,
        rr=df["rank"].where(hit, other=float("inf")).rdiv(1),
    ).groupby("chunker", sort=False)
    return pd.DataFrame(
        {
            "fragmentos": summary["n_fragmentos"].first(),
            "recall@1": summary["r1"].mean(),
            f"recall@{k}": summary["rk"].mean(),
            "MRR": summary["rr"].mean(),
            "sim_media_acierto": summary["sim_acierto"].mean(),
        }
    )


# --- 2. Agente ----------------------------------------------------------------


class ThrottledClient:
    """Cliente del LLM que respeta un máximo de peticiones por minuto.

    La capa gratuita de Flash-Lite admite 15 por minuto; la evaluación lanza
    cientos de peticiones seguidas, así que se espera cuando hace falta.
    """

    def __init__(self, client: GeminiClient, rpm: int = 12):
        """Envuelve un cliente con un límite de ``rpm`` peticiones/minuto."""
        self.client = client
        self.rpm = rpm
        self._calls: deque[float] = deque()

    @property
    def last_model(self) -> str | None:
        """Modelo que respondió a la última petición."""
        return self.client.last_model

    def _wait(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= self.rpm:
            time.sleep(60 - (now - self._calls[0]) + 0.5)
        self._calls.append(time.monotonic())

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Igual que :meth:`GeminiClient.generate`, con espera previa."""
        self._wait()
        return self.client.generate(*args, **kwargs)

    def generate_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Igual que :meth:`GeminiClient.generate_stream`, con espera previa."""
        self._wait()
        yield from self.client.generate_stream(*args, **kwargs)


def check_case(
    question: dict[str, Any], record: dict[str, Any], today: date
) -> dict[str, bool | None]:
    """Comprobaciones automáticas (sin LLM) de un caso.

    Args:
        question: Pregunta del conjunto de evaluación.
        record: Respuesta y llamadas a herramientas del agente.
        today: Fecha de ejecución, para resolver las fechas relativas.

    Returns:
        ``None`` cuando una comprobación no aplica al caso:

        - ``herramientas_ok``: usó las herramientas esperadas y, si no debía
          dar un pronóstico, no obtuvo ninguno.
        - ``tiempo_ok``: consultó el tiempo del lugar y la fecha correctos.
        - ``recuperado_ok``: algún fragmento recuperado contiene la evidencia.
        - ``cita_ok``: cita alguna de las páginas esperadas.
        - ``enlace_ok``: incluye un enlace de Google Maps.
    """
    calls = record["herramientas"]
    names = {call["name"] for call in calls}
    behaviour = question["comportamiento"]
    forecast = any(c["name"] == "get_weather" and c["ok"] for c in calls)

    tools_ok = set(question["herramientas"]) <= names
    if behaviour in NO_FORECAST:
        tools_ok = tools_ok and not forecast

    weather_ok = None
    if spec := question.get("tiempo"):
        dates = accepted_dates(spec["fecha"], today)
        weather_ok = any(
            c["name"] == "get_weather"
            and c["ok"]
            and c["args"].get("lugar") in spec["lugar"]
            and c["args"].get("fecha") in dates
            for c in calls
        )

    retrieved_ok = None
    if question["evidencia"] and "buscar_guia" in question["herramientas"]:
        texts = [
            fragment["texto"]
            for c in calls
            if c["name"] == "buscar_guia"
            for fragment in c["result"].get("fragmentos", [])
        ]
        retrieved_ok = any(contains_evidence(t, question["evidencia"]) for t in texts)

    citation_ok = None
    if behaviour == "responder_con_cita" and question["paginas"]:
        citation_ok = bool(cited_pages(record["respuesta"]) & set(question["paginas"]))

    link_ok = (
        "google.com/maps" in record["respuesta"] if behaviour == "dar_enlace" else None
    )

    return {
        "herramientas_ok": tools_ok,
        "tiempo_ok": weather_ok,
        "recuperado_ok": retrieved_ok,
        "cita_ok": citation_ok,
        "enlace_ok": link_ok,
    }


def results_path(label: str) -> Path:
    """Archivo de resultados de una ejecución del agente."""
    return RESULTS_DIR / f"agente_{label}.jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    """Lee los resultados guardados (lista vacía si no existen)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_records(records: list[dict[str, Any]], path: Path) -> None:
    """Guarda los resultados, uno por línea."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_agent_eval(
    questions: list[dict[str, Any]],
    label: str,
    client: Any = None,
    tools: ToolRegistry | None = None,
    force: bool = False,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Ejecuta el asistente sobre el conjunto de evaluación.

    Cada pregunta se responde en una conversación nueva (precedida de su
    historial, si lo tiene). Los resultados se guardan tras cada pregunta,
    de modo que una ejecución interrumpida se reanuda donde se quedó.

    Args:
        questions: Conjunto de evaluación.
        label: Nombre de la ejecución (p. ej. "LOW").
        client: Cliente del LLM (por defecto, Gemini con límite de ritmo).
        tools: Herramientas (por defecto, las del asistente).
        force: Repetir las preguntas ya respondidas.
        path: Archivo de resultados (por defecto, el de ``label``).

    Returns:
        Un registro por pregunta con la respuesta, las herramientas, las
        métricas de la traza y las comprobaciones automáticas.
    """
    path = path or results_path(label)
    records = [] if force else load_records(path)
    done = {record["id"] for record in records}
    client = client or ThrottledClient(GeminiClient())
    tools = tools or ToolRegistry()
    today = canary_today()

    for question in questions:
        if question["id"] in done:
            continue
        agent = TouristAgent(client=client, tools=tools, memory=ConversationMemory())
        # Herramientas de toda la conversación: en una pregunta de
        # seguimiento, los datos pueden venir de un turno anterior.
        calls: list[dict[str, Any]] = []
        for previous in question.get("historial", []):
            agent.chat(previous)
            calls += [
                asdict(c) | {"turno": "previo"} for c in agent.last_trace.tool_calls
            ]
        reply = agent.chat(question["pregunta"])
        trace = reply.trace
        calls += [asdict(c) | {"turno": "final"} for c in trace.tool_calls]
        record = {
            "id": question["id"],
            "tipo": question["tipo"],
            "pregunta": question["pregunta"],
            "historial": question.get("historial", []),
            "comportamiento": question["comportamiento"],
            "respuesta": reply.text,
            "herramientas": calls,
            "vueltas": trace.rounds,
            "prompt_tokens": trace.prompt_tokens,
            "output_tokens": trace.output_tokens,
            "thinking_tokens": trace.thinking_tokens,
            "primer_token_s": trace.first_token_s,
            "total_s": trace.total_s,
            "modelo": trace.model,
            "fin": trace.finish_reasons,
            "forzado": trace.forced_final,
            "error": trace.error,
            "fecha_ejecucion": today.isoformat(),
        }
        if trace.error:
            # No se guarda: la pregunta se repetirá en la siguiente ejecución.
            logger.warning("[%s] %s falló: %s", label, question["id"], trace.error)
            continue
        record["checks"] = check_case(question, record, today)
        records.append(record)
        save_records(records, path)
        logger.info(
            "[%s] %s respondida en %.1f s", label, question["id"], trace.total_s
        )
    return records


# --- 3. Juez ------------------------------------------------------------------


def judge_records(
    records: list[dict[str, Any]],
    path: Path,
    judge: Judge | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Añade el veredicto del juez a cada registro y los guarda.

    Args:
        records: Resultados de :func:`run_agent_eval`.
        path: Archivo donde se guardan.
        judge: Juez (por defecto, Flash-Lite con temperatura 0 y límite de
            ritmo).
        force: Volver a juzgar los registros que ya tienen veredicto.

    Returns:
        Los mismos registros con el campo ``juez`` (o ``juez_error``).
    """
    judge = judge or Judge(
        ThrottledClient(GeminiClient(replace(MODEL, temperature=0.0)))
    )
    for record in records:
        if "juez" in record and not force:
            continue
        try:
            record["juez"] = judge.evaluate(record).model_dump()
            record.pop("juez_error", None)
        except LLMError as err:
            record["juez_error"] = str(err)
        save_records(records, path)
    return records


# --- Resúmenes ------------------------------------------------------------------


def records_frame(records: list[dict[str, Any]], label: str = "") -> pd.DataFrame:
    """Tabla plana con una fila por pregunta (métricas, checks y juez)."""
    rows = []
    for record in records:
        verdict = record.get("juez") or {}
        rows.append(
            {
                "ejecucion": label,
                "id": record["id"],
                "tipo": record["tipo"],
                "comportamiento": record["comportamiento"],
                "vueltas": record["vueltas"],
                "herramientas": len(record["herramientas"]),
                "prompt_tokens": record["prompt_tokens"],
                "output_tokens": record["output_tokens"],
                "thinking_tokens": record["thinking_tokens"],
                "primer_token_s": record["primer_token_s"],
                "total_s": record["total_s"],
                **record["checks"],
                "fidelidad": verdict.get("fidelidad"),
                "relevancia": verdict.get("relevancia"),
                "comportamiento_ok": verdict.get("comportamiento_ok"),
            }
        )
    return pd.DataFrame(rows)


def agent_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Métricas medias por ejecución (las comprobaciones ignoran los N/A).

    Returns:
        Una fila por ejecución: tasas de acierto de las comprobaciones,
        notas medias del juez, latencias y tokens.
    """
    checks = ["herramientas_ok", "tiempo_ok", "recuperado_ok", "cita_ok", "enlace_ok"]
    grouped = frame.groupby("ejecucion", sort=False)
    summary = pd.DataFrame(
        {
            name: grouped[name].apply(lambda s: s.dropna().astype(float).mean())
            for name in [*checks, "comportamiento_ok"]
        }
    )
    summary["fidelidad"] = grouped["fidelidad"].mean()
    summary["relevancia"] = grouped["relevancia"].mean()
    summary["latencia_media_s"] = grouped["total_s"].mean()
    summary["latencia_p90_s"] = grouped["total_s"].quantile(0.9)
    summary["primer_token_medio_s"] = grouped["primer_token_s"].mean()
    summary["tokens_entrada_medios"] = grouped["prompt_tokens"].mean()
    summary["tokens_salida_medios"] = grouped["output_tokens"].mean()
    summary["vueltas_medias"] = grouped["vueltas"].mean()
    return summary


# --- 4. Validación del juez -------------------------------------------------------


def export_for_review(records: list[dict[str, Any]], path: Path) -> None:
    """Exporta respuestas y fuentes, sin las notas del juez, para revisarlas.

    La revisión debe hacerse sin ver las notas del juez para que sea
    independiente.
    """
    from src.judge import BEHAVIOURS, format_sources

    lines = ["# Respuestas para revisión independiente\n"]
    for record in records:
        lines += [
            f"## {record['id']} · {record['tipo']}",
            f"**Historial:** {' / '.join(record['historial']) or '(ninguno)'}",
            f"**Pregunta:** {record['pregunta']}",
            f"**Esperado:** {BEHAVIOURS[record['comportamiento']]}",
            "**Fuentes:**",
            "```",
            format_sources(record["herramientas"]),
            "```",
            "**Respuesta:**",
            "",
            record["respuesta"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def judge_agreement(
    records: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compara las notas del juez con una revisión de referencia.

    Criterio de fiabilidad, fijado antes de ver los resultados:

    - al menos un 80 % de acuerdo en fidelidad con una diferencia máxima de
      1 punto, y
    - ninguna alucinación de la referencia (fidelidad <= 3) aprobada por el
      juez (fidelidad >= 4).

    Args:
        records: Registros con el veredicto del juez.
        reference: Notas de referencia (``id``, ``fidelidad``,
            ``relevancia``, ``comportamiento_ok``).

    Returns:
        Tasas de acuerdo, casos en desacuerdo y si el juez es fiable.
    """
    judged = {r["id"]: r["juez"] for r in records if r.get("juez")}
    pairs = [(ref, judged[ref["id"]]) for ref in reference if ref["id"] in judged]
    if not pairs:
        return {"casos": 0, "fiable": False}

    def rate(condition: Callable[[dict, dict], bool]) -> float:
        return sum(condition(ref, jud) for ref, jud in pairs) / len(pairs)

    missed = [
        ref["id"]
        for ref, jud in pairs
        if ref["fidelidad"] <= 3 and jud["fidelidad"] >= 4
    ]
    within_one = rate(lambda r, j: abs(r["fidelidad"] - j["fidelidad"]) <= 1)
    return {
        "casos": len(pairs),
        "fidelidad_exacta": rate(lambda r, j: r["fidelidad"] == j["fidelidad"]),
        "fidelidad_±1": within_one,
        "relevancia_±1": rate(lambda r, j: abs(r["relevancia"] - j["relevancia"]) <= 1),
        "comportamiento_igual": rate(
            lambda r, j: r["comportamiento_ok"] == j["comportamiento_ok"]
        ),
        "alucinaciones_no_detectadas": missed,
        "desacuerdos": [
            {"id": ref["id"], "referencia": ref["fidelidad"], "juez": jud["fidelidad"]}
            for ref, jud in pairs
            if abs(ref["fidelidad"] - jud["fidelidad"]) > 1
        ],
        "fiable": within_one >= 0.8 and not missed,
    }


# --- Ejecución por línea de comandos ----------------------------------------------


def main() -> None:
    """Lanza los pasos de la evaluación desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluación del asistente")
    parser.add_argument(
        "--paso", choices=["recuperacion", "agente", "juez", "todo"], default="todo"
    )
    parser.add_argument("--thinking", default=MODEL.thinking_level)
    parser.add_argument(
        "--etiqueta",
        help="Nombre de la ejecución (por defecto, el nivel de razonamiento)",
    )
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "google_genai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    questions = load_questions()
    if args.paso in ("recuperacion", "todo"):
        df = evaluate_retrieval(questions)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / "recuperacion.csv", index=False)
        print(retrieval_summary(df).round(3).to_string())

    thinking = args.thinking.upper()
    label = args.etiqueta or thinking
    path = results_path(label)
    if args.paso in ("agente", "todo"):
        # Sin modelos de respaldo: si se agota la cuota, el experimento se
        # detiene en lugar de mezclar respuestas de otro modelo.
        config = replace(MODEL, thinking_level=thinking, fallback_models=())
        run_agent_eval(
            questions,
            label,
            client=ThrottledClient(GeminiClient(config)),
            force=args.forzar,
        )
    if args.paso in ("juez", "todo"):
        records = judge_records(load_records(path), path, force=args.forzar)
        print(agent_summary(records_frame(records, label)).round(2).T.to_string())


if __name__ == "__main__":
    main()
