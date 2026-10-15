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
