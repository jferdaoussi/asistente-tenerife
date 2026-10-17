# Despliegue en Render

La app (`app.py` y `src/`) se despliega como *Web Service* de Render, que
construye el código desde el repositorio de GitHub y lo vuelve a desplegar
con cada `git push`. El LLM no se aloja: la app llama a la API de Gemini con
la clave guardada como secreto del servicio.

## Crear el servicio

En el panel de Render: **New → Web Service**, elegir el repositorio
`asistente-tenerife` y rellenar:

| Campo | Valor |
|---|---|
| Language | Python 3 |
| Branch | `main` |
| Build Command | `pip install -r deploy/requirements.txt` |
| Start Command | `python app.py` |
| Instance Type | Free |

En **Environment Variables**:

| Variable | Valor |
|---|---|
| `PYTHON_VERSION` | `3.12` |
| `GEMINI_API_KEY` | Clave de Gemini (secreta) |
| `APP_USER` | Opcional: usuario de acceso a la demo |
| `APP_PASSWORD` | Opcional: contraseña de acceso a la demo (secreta) |

La demo desplegada no las define, así que es pública. Si se añaden las dos,
la app pide inicio de sesión. El puerto lo indica Render en `PORT` y la app
lo usa.

## Notas

- **Plan gratuito:** el servicio se duerme tras 15 minutos sin visitas y la
  primera visita tarda alrededor de un minuto en despertarlo.
- **Índice de la guía:** se construye al arrancar (20 fragmentos, una
  petición a la API de embeddings). El disco del plan gratuito no es
  persistente, así que se rehace en cada arranque.
- **Memoria:** la app ocupa unos 230 MB; la instancia gratuita tiene 512 MB.
- **Por qué no Hugging Face Spaces:** en octubre de 2026, alojar Spaces de
  Gradio en la CPU gratuita exige la suscripción PRO (error 402 al crear el
  Space).
