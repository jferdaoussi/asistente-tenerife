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
├── app.py          # App web (Gradio)
├── notebook/       # Notebook principal (asistente_tenerife.ipynb)
├── data/           # Guía turística (TENERIFE.pdf)
├── src/            # Código del asistente (módulos reutilizables)
├── eval/           # Conjunto de preguntas y resultados de la evaluación
├── tests/          # Tests automáticos (pytest)
├── docs/           # Decisiones técnicas e informe final
├── deploy/         # Despliegue de la app en Render
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

## Ejecutar el notebook

El notebook principal es `notebook/asistente_tenerife.ipynb`. Se entrega con
las salidas guardadas, así que los resultados se ven sin ejecutarlo.

Para ejecutarlo hay que usar como kernel el entorno virtual del proyecto
(`.venv`), que es donde están instaladas las dependencias:

- **VS Code:** *Select Kernel* → *Python Environments* → `.venv`.
- **Jupyter:** registra el entorno una vez y elige *Python (asistente-tenerife)*
  en *Kernel → Change kernel*:

```bash
python -m ipykernel install --user --name asistente-tenerife --display-name "Python (asistente-tenerife)"
jupyter lab notebook/asistente_tenerife.ipynb
```

La primera celda añade la raíz del repositorio a la ruta de importación, de
modo que el notebook usa los módulos de `src/`. Sin clave de Gemini, las
celdas que llaman a la API muestran un aviso y el resto funciona.

## App web

![App web del asistente](docs/img/app.jpg)

- **Chat en streaming**: la respuesta aparece palabra a palabra.
- **Fuentes desplegables** bajo cada respuesta: los fragmentos de la guía
  citados (con su texto), los datos externos (tiempo, ruta) y, aparte, lo que
  la búsqueda recuperó pero la respuesta no usó.
- **Panel de observabilidad**: latencia, tokens, vueltas del agente, uso de
  la memoria frente a su presupuesto y cada herramienta con sus argumentos,
  su resultado y su duración.
- **Ejemplos** que se preguntan con un clic y una conversación propia por
  visitante.

Para arrancarla en local (con la clave en `.env`):

```bash
python app.py
```

y abrir http://127.0.0.1:7860.

### Despliegue

La app está desplegada en Render como *Web Service* gratuito, con acceso por
usuario y contraseña (`APP_USER` y `APP_PASSWORD`) y la clave de Gemini como
secreto del servicio. Los pasos y la configuración están en
[`deploy/README.md`](deploy/README.md).

Límites de la demo (`AppConfig` en `src/config.py`): 20 mensajes por
conversación, 500 caracteres por mensaje y 2 peticiones a la vez.
