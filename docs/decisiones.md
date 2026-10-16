# Registro de decisiones técnicas

Decisiones de diseño tomadas durante el proyecto, con su contexto, las
alternativas consideradas y su justificación. Es la base de la sección
"Decisiones técnicas" del informe final y del notebook.

Formato de cada entrada: **contexto → decisión → alternativas → por qué →
consecuencias**.

---

## D1 · Proveedor de LLM: Google Gemini (capa gratuita)

- **Contexto:** el enunciado pide un LLM comercial accesible por API ("por
  ejemplo, OpenAI o Gemini"); no limita a esos dos.
- **Decisión:** Google Gemini con el SDK oficial `google-genai`.
- **Alternativas:** OpenAI (de pago); DeepSeek V4.1 Flash gratuito a través de
  OpenCode (promoción de un tercero); API oficial de DeepSeek (de pago, muy
  barata).
- **Por qué:** capa gratuita oficial y estable; chat, embeddings y visión con
  la misma clave; documentación oficial de function calling. La promoción de
  DeepSeek no tenía fecha de fin (una anterior desapareció sin aviso), no
  aclaraba su uso fuera de OpenCode y DeepSeek no ofrece embeddings.
- **Consecuencias:** límites de uso de la capa gratuita (ver D2) y Google
  puede usar los datos enviados para mejorar sus productos. Todas las llamadas
  al LLM están aisladas en `src/llm.py` para poder cambiar de proveedor
  (plan B: API de pago de DeepSeek).

## D2 · Modelo principal: `gemini-3.5-flash-lite`

- **Contexto:** límites de la capa gratuita medidos en AI Studio (15/10/2026):
  Gemini 3.6 Flash y 3.8 Flash permiten 5 peticiones/min y **20 al día**;
  Gemini 3.5 Flash-Lite, 15/min y **500 al día**. Un turno con herramientas
  gasta 2-3 peticiones y la evaluación necesita unas 100.
- **Decisión:** `gemini-3.5-flash-lite` como principal; `gemini-3.6-flash` y
  `gemini-3.8-flash` como respaldo automático ante errores 429/5xx.
- **Alternativas:** `gemini-3.8-flash` (el más reciente) o `gemini-3.6-flash`.
- **Por qué:** con 20 peticiones al día la evaluación es inviable. Además,
  `gemini-3.8-flash` devolvía errores 503 por alta demanda (hasta 245 s de
  espera). Flash-Lite responde sin gastar tokens de razonamiento y basta
  para redactar a partir de fragmentos ya recuperados.
- **Consecuencias:** posible menor calidad de redacción que Flash (a medir en
  la evaluación); hacen falta caché y pausas al evaluar.

## D3 · Base de datos vectorial: Chroma

- **Contexto:** el enunciado sugiere FAISS o Chroma ("u otro").
- **Decisión:** Chroma en local, persistente en disco, con distancia coseno.
- **Alternativas:** FAISS.
- **Por qué:** Chroma guarda vectores, texto y metadatos juntos y permite
  filtrar por metadatos (zona, categoría) durante la búsqueda. FAISS es solo
  un índice de búsqueda: los metadatos y la persistencia habría que
  gestionarlos a mano. Con unos 20 fragmentos, la velocidad no es un factor.
- **Consecuencias:** hay que configurar explícitamente la distancia coseno
  (Chroma usa L2 por defecto) y convertir distancia en similitud
  (similitud = 1 − distancia).

## D4 · Chunking con funciones propias guiadas por la estructura de la guía

- **Contexto:** el chunking decide qué unidad de texto se convierte en un
  vector; es la decisión con más impacto en la calidad de la recuperación.
  La guía tiene una estructura muy marcada: zonas ("ZONA NORTE", "ZONA SUR",
  "OTRAS VAINAS"), sitios ("•"), puntos de interés ("o") y restaurantes
  ("▪").
- **Decisión:** implementar el chunking con funciones propias
  (`src/chunking.py`) en lugar de usar una librería: un fragmento por sitio,
  con cabecera de contexto ("Zona Norte › La Orotava") y metadatos (sitio,
  zona, categoría, páginas). Las secciones de más de 1.200 caracteres se
  dividen entre puntos de interés, nunca a mitad de frase.
- **Alternativas:** las librerías de referencia del sector:
  - LangChain `RecursiveCharacterTextSplitter` / `CharacterTextSplitter`
    (corte por tamaño con solapamiento; el estándar de facto en prototipos).
  - LlamaIndex `SentenceSplitter`.
  - Unstructured `chunk_by_title` y Docling `HybridChunker` (corte por
    estructura, la misma idea que la nuestra, pero genérica).
  - Chunking semántico (corte donde cambia el tema según los embeddings).
- **Por qué:**
  1. Ninguna librería genérica conoce las marcas propias de esta guía ("•"
     = sitio nuevo, "o Zona Norte" = subzona); habría que reescribir la misma
     lógica encima de ellas.
  2. Cada fragmento trata un único sitio, lo que da citas limpias y permite
     filtrar por zona y categoría.
  3. Unas 100 líneas sin cajas negras: se entiende, se prueba y se defiende.
  4. Evita dependencias pesadas (LangChain) para una tarea pequeña.
- **Cómo se valida:** se compara con dos alternativas, con el mismo tamaño
  (800 caracteres) y solapamiento (150):
  - `chunk_fixed`: tamaño fijo, propia (equivale al `CharacterTextSplitter`
    de LangChain).
  - `chunk_recursive`: el `RecursiveCharacterTextSplitter` **real** de
    LangChain (`langchain-text-splitters`), la referencia del sector.

  Análisis estructural (15/10/2026, sin embeddings):

  | Estrategia | Fragmentos | Mezclan 2 sitios | Empiezan a mitad de frase | Con sitio y zona |
  |---|---|---|---|---|
  | Por sitio (propia) | 20 | 0 | 0 | 20 |
  | Tamaño fijo (propia) | 24 | 12 | 19 | 0 |
  | LangChain recursivo | 27 | 9 | 10 | 0 |

  La comparación definitiva se hace en la evaluación, con las mismas
  preguntas y métricas de recuperación para las tres. Caso a vigilar: en
  "¿Dónde comprar gofio?" el chunking por sitio recupera primero "Sitios
  para comer", mientras que un fragmento de LangChain contiene justo la
  frase del gofio en La Orotava.
- **Consecuencias:** el chunker depende del formato de esta guía; con otro
  documento habría que adaptar las marcas o recurrir a una librería genérica
  (limitación para el informe).

## D5 · Embeddings: Gemini Embedding 2 a 768 dimensiones, asimétricos

- **Decisión:** `gemini-embedding-2`, recortado a 768 dimensiones, con
  prefijos distintos para fragmentos (`title: ... | text: ...`) y preguntas
  (`task: search result | query: ...`).
- **Alternativas:** 3072 dimensiones (nativo); `gemini-embedding-001`;
  modelos locales (`sentence-transformers`).
- **Por qué:** multilingüe (la guía está en español), gratuito y con la misma
  clave. 768 es uno de los tamaños recomendados por Google: casi la misma
  calidad con 4 veces menos espacio (técnica Matryoshka). Los prefijos
  asimétricos indican al modelo qué texto es pregunta y cuál documento.

## D6 · Umbral de similitud: 0,62, calibrado con datos

- **Contexto:** la búsqueda vectorial siempre devuelve algo, aunque no tenga
  relación con la pregunta.
- **Decisión:** descartar fragmentos con similitud menor que 0,62.
- **Por qué:** medido el 15/10/2026: la mejor similitud de 15 preguntas que
  la guía sí responde fue de 0,624 a 0,818; la de 10 preguntas ajenas, de
  0,553 a 0,684. Con 0,62 pasan las 15 de la guía y se rechazan 9 de 10.
- **Consecuencias:** "¿Qué playas hay en Gran Canaria?" (0,684) supera el
  umbral; la segunda barrera es el system prompt del LLM ("solo Tenerife y
  solo lo que diga la guía").

## D7 · Limpieza de marcas de enlace del PDF

- **Contexto:** el PDF se exportó desde Word con "Imprimir a PDF" y **no
  conserva ningún hipervínculo**; las marcas "[vídeo - ubicación]", "[web]"
  o "[Tripadvisor]" son texto sin URL.
- **Decisión:** eliminarlas al cargar el texto (`src/loader.py`).
- **Por qué:** no aportan información y harían que el LLM prometiera enlaces
  que no existen.
- **Consecuencias:** quedan frases como "aparcar por aquí" sin referencia
  (limitación de la fuente). Para indicar cómo llegar a un sitio se usará un
  enlace de búsqueda de Google Maps generado a partir de su nombre.

## D8 · `get_weather`: Open-Meteo, solo pronóstico, por lugar de la guía

- **Contexto:** el enunciado exige una función `get_weather(fecha)` con
  esquema, validación, manejo de errores y registro de intentos. Límites
  reales de Open-Meteo medidos el 16/10/2026: el pronóstico llega hasta hoy +
  15 días (después, HTTP 400) y la misma API devuelve datos de días pasados,
  que no coinciden con los de su API de histórico (ayer: 26,7 °C frente a
  28,9 °C de máxima).
- **Decisiones:**
  1. **Open-Meteo**: gratuita, sin clave y sin registro.
  2. **Solo de hoy a hoy + 15 días.** Las fechas pasadas son un error
     controlado: la herramienta sirve para planificar, y ofrecer el pasado
     obligaría a elegir entre dos fuentes que no coinciden.
  3. **Parámetro `lugar` con lista cerrada de 16 lugares de la guía.**
     Tenerife tiene microclimas: el 17/10/2026 se pronosticaban 7 °C de
     máxima en la cumbre del Teide y 28 °C en Punta de Teno. Coordenadas fijas
     en el código (geocodificación de Open-Meteo o punto de visita de la
     guía), sin geocodificar en cada llamada.
  4. **Validación en nuestro código antes de llamar a la API** (Pydantic):
     formato `AAAA-MM-DD`, rango de fechas y lugar conocido; se aceptan
     variantes sin tildes ni mayúsculas. Ahorra peticiones y da mensajes
     claros en español ("disponible a partir del ...").
  5. **Un único modelo Pydantic** (`WeatherRequest`) genera el JSON Schema
     que ve el LLM y valida los argumentos: esquema y validación no pueden
     desincronizarse.
  6. **Los errores se devuelven al LLM como datos** (`{"ok": false, "error",
     "mensaje"}`), nunca como excepciones, para que pueda explicarlos,
     corregir sus argumentos (p. ej. "Mazca" → "Masca" con la lista de lugares
     válidos) o preguntar al usuario.
  7. **Salida filtrada a 13 datos útiles para un turista**: cielo (código WMO
     traducido a texto), temperaturas, sensación térmica, lluvia, viento y
     rachas, UV, humedad, amanecer y atardecer; unidades en el nombre de cada
     campo. Quien lee la salida es el LLM: menos campos implican menos tokens
     y menos ruido. Aviso de fiabilidad a partir de 7 días.
  8. **Reintento solo ante fallos transitorios** (timeout, conexión, 429,
     5xx), con 2 intentos y 1 s de espera; un 4xx no se reintenta.
  9. **Registro JSON Lines** (`logs/tools.jsonl`): una línea por intento con
     argumentos, etapa (validación o API), número de intento, resultado o
     error, y latencia.
- **Quién resuelve las fechas relativas:** el LLM convierte "el sábado" en
  `2026-10-17`; para ello el system prompt le indicará la fecha de hoy en
  Canarias (Fase 4). El código no se fía y lo valida.
- **Consecuencias:** no hay tiempo para sitios fuera de la lista, como pueblos
  no mencionados en la guía (el LLM elegirá el lugar más cercano). La calima
  (polvo del Sáhara), muy relevante en Canarias, requiere la API de calidad
  del aire de Open-Meteo: mejora futura.

## D9 · Agente propio: búsqueda como herramienta, bucle limitado y memoria en tres niveles

- **Contexto:** el asistente debe combinar la guía (RAG), el tiempo y la
  conversación de varios turnos con un control de tokens.
- **Decisiones:**
  1. **Bucle de herramientas propio** (`src/agent.py`) en lugar de la
     llamada automática a funciones del SDK o de un framework de agentes
     (LangChain, LlamaIndex): permite validar argumentos, registrar cada
     llamada, limitar las vueltas y entender el flujo completo. Son unas 100
     líneas.
  2. **La búsqueda en la guía es una herramienta (`buscar_guia`)**, no un
     paso fijo antes de cada respuesta. Ante "hola" o "gracias" no se
     busca, y en preguntas de seguimiento ("¿y dónde comer allí?") el propio
     LLM redacta una consulta autónoma ("dónde comer en Masca"), lo que
     sustituye al paso de reescritura de consultas previsto. **Riesgo:** que
     el LLM responda de memoria sin buscar; se mitiga con la regla 1 del
     system prompt y se mide en la evaluación.
  3. **Máximo de 4 vueltas** por turno (una vuelta = una petición al LLM, que
     puede pedir varias herramientas en paralelo). Al alcanzarlo se hace una
     última petición con las herramientas desactivadas (`mode="NONE"`) para
     obligar a responder con lo obtenido. Protege la cuota (15 peticiones/min)
     ante bucles por fallos repetidos.
  4. **Memoria en tres niveles** (`src/memory.py`): último turno completo con
     resultados de herramientas; turnos anteriores solo con pregunta y
     respuesta; por encima de 6.000 tokens de prompt (medidos con el recuento
     que devuelve la API), los turnos antiguos se resumen con el LLM en un
     párrafo con el perfil del viajero, que pasa al system prompt. Si el
     resumen falla, se conserva el historial.
  5. **System prompt regenerado en cada turno** (`src/prompts.py`) con la
     fecha de hoy en Canarias y un calendario de los 15 días siguientes
     ("sábado 17/10/2026 = 2026-10-17"): el LLM busca la fecha en una tabla
     en lugar de hacer aritmética de calendario. Días de la semana con una
     lista propia (`strftime` depende del idioma del sistema).
  6. **Respuestas en streaming** y una **traza por turno** (vueltas,
     herramientas, fuentes, tokens, latencias, motivo de fin) para el
     notebook, la evaluación y el panel de la app.
- **Ajustes tras la primera conversación real (16/10/2026):**
  - El LLM citaba los datos del tiempo con una página de la guía → regla:
    "según Open-Meteo", sin cita de página.
  - Ante "el tiempo en Tenerife" elegía Santa Cruz por su cuenta (la
    descripción del parámetro decía "elige el más cercano") → descripción y
    regla 5 reescritas: si no hay sitio ni zona, preguntar.
  - Añadía descripciones de su conocimiento general a sitios que la guía solo
    nombra (Las Teresitas: "arena dorada, dique") → regla 2 reforzada; en dos
    repeticiones posteriores, respuestas fieles a los fragmentos.
  - Gemini devolvió en ocasiones una respuesta vacía (`STOP` sin contenido) →
    la vuelta se reintenta una vez; si persiste, mensaje de disculpa.
- **Consecuencias:** latencia de 3-4 s en respuestas sin herramientas y de
  12-30 s con 2-3 vueltas (a medir en la evaluación, junto con el nivel de
  razonamiento LOW frente a MINIMAL). La fidelidad depende de que el modelo
  siga las reglas: se medirá con un juez LLM.
