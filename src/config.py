"""Configuración central del asistente.

Reúne en un único lugar las rutas del proyecto, los parámetros del modelo,
los del RAG y los de la memoria conversacional, para que sean visibles y
fáciles de ajustar. La API key se lee del entorno (archivo ``.env``) y nunca
se escribe en el código.

Ejecutar ``python -m src.config`` muestra la configuración activa y comprueba
que la clave está disponible, sin imprimirla.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GUIDE_PATH = DATA_DIR / "TENERIFE.pdf"
CHROMA_DIR = ROOT_DIR / "chroma_db"
LOG_DIR = ROOT_DIR / "logs"

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class ModelConfig:
    """Parámetros de generación del LLM.

    Attributes:
        model: Identificador del modelo de Gemini.
        temperature: Aleatoriedad del muestreo. Baja para que el asistente se
            ciña a la guía en lugar de inventar.
        top_p: Nucleus sampling; solo se muestrea entre los tokens que suman
            esta probabilidad acumulada.
        max_output_tokens: Longitud máxima de cada respuesta.
        thinking_level: Cuánto "razona" el modelo antes de responder
            (MINIMAL, LOW, MEDIUM, HIGH). MINIMAL según la evaluación:
            misma calidad que LOW con un tercio de la latencia (D12).
        fallback_models: Modelos de respaldo que se prueban, en orden, si el
            principal está saturado (errores 429/5xx).
        retry_attempts: Intentos por modelo antes de pasar al siguiente.
        timeout_s: Tiempo máximo de espera de cada petición (en streaming,
            entre fragmentos), en segundos.

    Elección del modelo principal (pruebas del 15/10/2026, capa gratuita):

    - ``gemini-3.6-flash`` y ``gemini-3.8-flash`` solo permiten 20 peticiones
      al día y 5 por minuto; un turno con herramientas gasta 2-3 peticiones,
      lo que hace inviable la evaluación. Además, ``gemini-3.8-flash``
      devolvía errores 503 por alta demanda (hasta 245 s de espera).
    - ``gemini-3.5-flash-lite`` permite 500 peticiones al día y 15 por
      minuto, responde sin gastar tokens de razonamiento y basta para
      redactar respuestas a partir de fragmentos ya recuperados.

    Los modelos Flash quedan como respaldo si se agota la cuota de Flash-Lite.
    """

    model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    temperature: float = 0.3
    top_p: float = 0.95
    max_output_tokens: int = 1024
    thinking_level: str = "MINIMAL"
    fallback_models: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.8-flash")
    retry_attempts: int = 2
    timeout_s: int = 30


@dataclass(frozen=True)
class RAGConfig:
    """Parámetros de la recuperación semántica.

    Attributes:
        embedding_model: Modelo que convierte texto en vectores.
        embedding_dim: Dimensión de los vectores generados.
        collection_name: Nombre de la colección en Chroma.
        top_k: Número de fragmentos que se recuperan por consulta.
        min_similarity: Similitud mínima para considerar relevante un
            fragmento; por debajo, la guía "no tiene" la respuesta.
        chunk_max_chars: Tamaño máximo de un fragmento en el chunking por
            sitio (unos 300 tokens). Las secciones más largas se dividen por
            sus puntos de interés.
        fixed_chunk_size: Tamaño de los fragmentos en el chunking de tamaño
            fijo, que se usa como referencia en la evaluación.
        fixed_chunk_overlap: Solapamiento entre fragmentos consecutivos del
            chunking de tamaño fijo.
        embed_batch_size: Textos que se envían por petición de embeddings.

    Calibración de ``min_similarity``:

    - 15/10/2026, con 15 preguntas de la guía (mejor similitud de 0,624 a
      0,818) y 10 ajenas (de 0,553 a 0,684): se fijó 0,62.
    - 16/10/2026, evaluación con 36 preguntas: con 0,62 se descartaban
      fragmentos correctos de consultas cortas que escribe el agente
      ("pegarse un chuso": 0,598 en primera posición; "comprar gofio": La
      Orotava con 0,614 en tercera). Se baja a 0,55: solo descarta lo
      claramente ajeno, y la relevancia fina la decide el LLM, que ya elige
      cuándo buscar y qué usar de cada fragmento (ver D6 en
      docs/decisiones.md).
    """

    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768
    collection_name: str = "guia_tenerife"
    top_k: int = 4
    min_similarity: float = 0.55
    chunk_max_chars: int = 1200
    fixed_chunk_size: int = 800
    fixed_chunk_overlap: int = 150
    embed_batch_size: int = 50


@dataclass(frozen=True)
class MemoryConfig:
    """Parámetros del historial conversacional.

    La API del LLM no guarda estado: en cada petición se reenvía el historial.
    Para acotar su tamaño se aplican tres niveles:

    1. El último turno se conserva completo, con los resultados de las
       herramientas (fragmentos de la guía, datos del tiempo).
    2. Los ``keep_recent_turns`` turnos más recientes conservan solo el texto
       de la pregunta y de la respuesta: los resultados de herramientas
       antiguos ocupan mucho y se pueden volver a pedir.
    3. Si el prompt supera ``max_history_tokens``, los turnos anteriores se
       resumen con el LLM en un párrafo con el perfil del viajero (destino,
       fechas, con quién viaja, intereses) y lo ya recomendado.

    Attributes:
        max_history_tokens: Presupuesto de tokens del prompt (instrucciones,
            herramientas e historial). Se mide con el recuento que devuelve
            la propia API en cada respuesta.
        keep_recent_turns: Turnos recientes que se conservan literalmente.
        summary_max_words: Longitud máxima del resumen de turnos antiguos.
    """

    max_history_tokens: int = 6000
    keep_recent_turns: int = 3
    summary_max_words: int = 120


@dataclass(frozen=True)
class AgentConfig:
    """Parámetros del bucle del agente.

    Attributes:
        max_rounds: Vueltas máximas del bucle (cada vuelta es una petición al
            LLM, que puede pedir varias herramientas a la vez). Al alcanzarlo,
            se fuerza una respuesta final sin herramientas. Evita bucles por
            fallos repetidos que agotarían la cuota (15 peticiones/min).
    """

    max_rounds: int = 4


@dataclass(frozen=True)
class WeatherConfig:
    """Parámetros de la herramienta ``get_weather`` (API de Open-Meteo).

    Attributes:
        api_url: Endpoint de pronóstico de Open-Meteo (gratuito, sin clave).
        timezone: Zona horaria para calcular "hoy" y las horas de sol.
        max_days_ahead: Último día con pronóstico, contado desde hoy. La API
            rechaza fechas posteriores (comprobado el 16/10/2026: hoy+15).
        reliable_days: A partir de este día se avisa de que el pronóstico es
            orientativo.
        timeout_s: Tiempo máximo de espera de cada petición, en segundos.
        max_attempts: Intentos ante errores transitorios (timeout, 5xx).
        retry_delay_s: Espera entre intentos, en segundos.
    """

    api_url: str = "https://api.open-meteo.com/v1/forecast"
    timezone: str = "Atlantic/Canary"
    max_days_ahead: int = 15
    reliable_days: int = 7
    timeout_s: float = 10.0
    max_attempts: int = 2
    retry_delay_s: float = 1.0


MODEL = ModelConfig()
RAG = RAGConfig()
MEMORY = MemoryConfig()
AGENT = AgentConfig()
WEATHER = WeatherConfig()
TOOL_LOG_PATH = LOG_DIR / "tools.jsonl"


def get_api_key() -> str:
    """Devuelve la API key de Gemini leída del entorno.

    Returns:
        La clave definida en la variable ``GEMINI_API_KEY``.

    Raises:
        RuntimeError: Si la clave no está definida o sigue con el valor de
            ejemplo de ``.env.example``.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "pon-aqui-tu-clave":
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Copia .env.example como .env y pega tu "
            "clave de https://aistudio.google.com/apikey"
        )
    return key


if __name__ == "__main__":
    configs = (
        ("Modelo", MODEL),
        ("RAG", RAG),
        ("Memoria", MEMORY),
        ("Agente", AGENT),
        ("Tiempo", WEATHER),
    )
    for name, cfg in configs:
        print(f"[{name}]")
        for key, value in asdict(cfg).items():
            print(f"  {key} = {value}")
    print(f"[Guía] {GUIDE_PATH} (existe: {GUIDE_PATH.exists()})")
    try:
        get_api_key()
        print("[API key] encontrada en el entorno")
    except RuntimeError as err:
        print(f"[API key] {err}")
