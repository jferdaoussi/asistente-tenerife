# Asistente turístico de Tenerife

Asistente conversacional que responde preguntas sobre Tenerife a partir de una
guía turística en PDF. Combina:

- **RAG**: búsqueda semántica sobre la guía, con citas de la fuente.
- **Diálogo multiturno**: memoria de la conversación con control de tokens.
- **Function calling**: consulta de la previsión del tiempo (`get_weather`).

Proyecto final de la asignatura *Large Language Models* del Máster en IA,
Cloud Computing y DevOps (Pontia).

> 🚧 En desarrollo. Este README se completará con la arquitectura, los
> resultados de la evaluación y la guía de uso.

## Stack

| Pieza | Elección |
|---|---|
| LLM | Google Gemini (`gemini-3.5-flash-lite`, con respaldo automático a modelos Flash) mediante el SDK `google-genai` |
| Embeddings | Gemini Embedding 2 |
| Base de datos vectorial | Chroma (local, persistente) |
| Previsión del tiempo | Open-Meteo (sin API key) |

## Estructura

```
asistente-tenerife/
├── data/           # Guía turística (TENERIFE.pdf)
├── src/            # Código del asistente (módulos reutilizables)
├── notebooks/      # Notebook principal con la demostración
├── eval/           # Conjunto de preguntas y resultados de la evaluación
├── tests/          # Tests automáticos (pytest)
├── docs/           # Informe final
├── requirements.txt
└── .env.example    # Plantilla de configuración de la API key
```

## Instalación

Requiere **Python 3.12**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Con [uv](https://docs.astral.sh/uv/) es más rápido:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Configuración de la API key

1. Consigue una clave gratuita en [Google AI Studio](https://aistudio.google.com/apikey).
2. Copia `.env.example` como `.env` y pega la clave en `GEMINI_API_KEY`.
3. Comprueba la configuración:

```bash
python -m src.config
```

El archivo `.env` está en `.gitignore` y nunca se sube al repositorio.
