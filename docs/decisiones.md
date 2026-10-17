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

  Análisis estructural (sin embeddings; reproducible en la sección 3 del
  notebook: "mezcla" = aparece una marca de sitio "•" dentro del fragmento;
  "a mitad de frase" = empieza en minúscula y no es una viñeta):

  | Estrategia | Fragmentos | Mezclan 2 sitios | Empiezan a mitad de frase | Con sitio y zona |
  |---|---|---|---|---|
  | Por sitio (propia) | 20 | 0 | 0 | 20 |
  | Tamaño fijo (propia) | 24 | 12 | 19 | 0 |
  | LangChain recursivo | 27 | 9 | 1 | 0 |

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

## D6 · Umbral de similitud: 0,62 → 0,55, recalibrado con la evaluación

- **Contexto:** la búsqueda vectorial siempre devuelve algo, aunque no tenga
  relación con la pregunta.
- **Decisión inicial (15/10/2026):** descartar fragmentos con similitud menor
  que 0,62. La mejor similitud de 15 preguntas que la guía sí responde fue de
  0,624 a 0,818; la de 10 preguntas ajenas, de 0,553 a 0,684. Con 0,62
  pasaban las 15 de la guía y se rechazaban 9 de 10.
- **Revisión (16/10/2026, evaluación con 36 preguntas):** el umbral
  descartaba fragmentos correctos en las consultas cortas que redacta el
  agente. En "pegarse un chuso", el fragmento correcto salía primero con
  0,598; en "comprar gofio", La Orotava salía tercera con 0,614 y los cuatro
  resultados quedaban por debajo de 0,62. El agente respondía "la guía no lo
  incluye" y, en el caso del gofio, inventó que la guía mencionaba platos que
  no aparecen. **Se baja a 0,55.**
- **Por qué:** con la búsqueda como herramienta (D9), el LLM ya decide cuándo
  buscar y qué usar de cada fragmento; el umbral solo tiene que descartar lo
  claramente ajeno. Un umbral alto provoca falsos "no está en la guía", que
  son peores que recibir algún fragmento poco relacionado.
- **Consecuencias:** las preguntas ajenas casi siempre devuelven fragmentos;
  la barrera frente a ellas es el system prompt ("solo Tenerife y solo lo que
  diga la guía"), cuya eficacia se mide en la evaluación (fuera de dominio y
  prompt injection). Los datos de calibración muestran que en este corpus las
  similitudes están muy comprimidas (0,55-0,82), así que un umbral fijo
  discrimina poco.
- **Resultado (antes → después, junto con la regla 8 de "cómo llegar"):**
  evidencia recuperada 87,5 % → 95,8 %; herramienta correcta 97 % → 100 %;
  cita correcta 87 % → 91 %. Se corrigen los tres fallos sistemáticos (g01
  ya no inventa, g02 da la indicación de la guagua, g20 responde). El
  comportamiento global queda igual (94,4 %) por dos fallos nuevos sin
  relación con el cambio (g12 añade un dato externo; m02 vuelve a preguntar
  la zona pese al historial), que muestran la variabilidad entre ejecuciones
  del LLM. La latencia media sube de 12 a 15 s (búsqueda extra y más
  fragmentos en el contexto).

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
  - Tras "nos alojamos en el Puerto de la Cruz", a "¿qué tiempo hará
    mañana?" volvía a preguntar la zona (también en m02 de la evaluación):
    la regla 5 decía "si no indica lugar y la conversación no lo deja claro,
    pregunta" y el modelo aplicaba solo la segunda mitad. Se reescribió en
    positivo y por orden: sitio del usuario → sitio de la conversación →
    solo si no hay ninguno, preguntar. Comprobado en la misma conversación.
- **Consecuencias:** latencia de 3-4 s en respuestas sin herramientas y de
  12-30 s con 2-3 vueltas (a medir en la evaluación, junto con el nivel de
  razonamiento LOW frente a MINIMAL: ver D12). La fidelidad depende de que el modelo
  siga las reglas: se medirá con un juez LLM.

## D10 · Evaluación con conjunto fijo, evidencia literal y juez LLM validado

- **Contexto:** hay que medir la calidad del RAG, del uso de herramientas y
  del comportamiento, y comparar alternativas (chunkers, razonamiento) con
  la cuota gratuita (15 peticiones/min, 500/día).
- **Decisiones:**
  1. **Conjunto fijo de 36 preguntas** (`eval/preguntas.jsonl`): 20 que
     responde la guía, 3 que no cubre (incluida la trampa del horario del
     Siam Park), 2 fuera de dominio, 4 del tiempo (fechas relativas que se
     resuelven al ejecutar), 2 combinadas, 1 de ruta, 2 de prompt injection
     y 2 de seguimiento.
  2. **Frase de evidencia literal** en cada pregunta de la guía ("molino de
     Chano"): un fragmento es correcto si la contiene. Así, recall@k y MRR son
     comparables entre chunkers que trocean distinto. Las páginas esperadas
     se calculan buscando la frase en el PDF.
  3. **Comprobaciones automáticas sin LLM**: herramienta correcta, lugar y
     fecha del tiempo, evidencia recuperada, página citada y enlace.
  4. **Juez LLM** (Flash-Lite, temperatura 0, salida JSON con esquema) que
     lista cada afirmación y si las fuentes la respaldan antes de puntuar
     fidelidad, relevancia y comportamiento.
  5. **Validación del juez** con una revisión independiente de las 36
     respuestas hecha por Claude (un modelo de otra familia y más potente),
     sin ver las notas del juez. Criterio fijado de antemano: acuerdo en
     fidelidad (±1) ≥ 80 % y ninguna alucinación de la referencia aprobada.
     - Primera versión: 94 % de acuerdo, pero aprobó la alucinación de g01
       (no listó la frase "la guía menciona papas arrugadas", escondida junto
       a una negativa) → **no fiable**.
     - Plan B: prompt corregido (las afirmaciones sobre lo que la guía
       menciona cuentan aunque vayan junto a una negativa) y arnés corregido
       (las fuentes incluyen herramientas de turnos anteriores). Resultado:
       100 % de acuerdo ±1, 94 % exacto, 97 % en comportamiento, ninguna
       alucinación aprobada → **fiable**. Sesgo residual: algo indulgente con
       imprecisiones menores (da 5 donde la referencia da 4).
  6. **Resultados en caché** (`eval/resultados/`) y reanudables pregunta a
     pregunta; un cliente con límite de 12 peticiones/min y caché en disco de
     los vectores de las preguntas.
- **Consecuencias:** la validación del juez se hizo sobre las mismas
  respuestas con las que se corrigió su prompt; se comprueba fuera de muestra
  con las respuestas de las ejecuciones siguientes.

## D11 · Claves de la API: cada ejecutor usa la suya; la app, un secreto del servidor

- **Contexto:** el enunciado pide gestionar la clave con variables de
  entorno o un gestor seguro y que el prototipo sea reproducible, pero no
  indica con qué clave se ejecuta al corregir.
- **Decisión:**
  - **Notebook y código:** cada persona usa su propia clave gratuita de
    Gemini en `.env` (plantilla en `.env.example`; instrucciones en el
    README). La clave del autor no se comparte ni se incluye en el repo.
  - **Notebook entregado con las salidas guardadas**, para que los
    resultados se vean sin ejecutarlo; sin clave, las celdas que llaman a la
    API muestran un aviso en lugar de fallar.
  - **App web desplegada:** usa la clave del autor como secreto del servidor
    (nunca visible para quien la usa), con límites de uso.
- **Alternativas descartadas:** clave en el repositorio (quedaría en el
  historial de git y Google revoca las claves que detecta expuestas);
  enviar la clave al profesor (cuota compartida y ligada a la cuenta del
  autor).

## D12 · Razonamiento del modelo: MINIMAL en lugar de LOW

- **Contexto:** Flash-Lite permite elegir cuánto razona antes de responder.
  El agente empezó con LOW; la latencia (10-30 s en turnos con
  herramientas) era el punto más débil del asistente.
- **Experimento:** las 36 preguntas de la evaluación (D10) con LOW y con
  MINIMAL, sin modelos de respaldo y en la misma franja horaria (17/10/2026,
  12:25-12:45), para que la carga de la API sea comparable.

  | Métrica | LOW | MINIMAL |
  |---|---|---|
  | Herramientas correctas | 100 % | 100 % |
  | Evidencia recuperada | 96 % | 96 % |
  | Página citada | 91 % | 96 % |
  | Comportamiento (juez) | 97 % | 100 % |
  | Fidelidad / relevancia (juez, 1-5) | 4,83 / 4,94 | 4,97 / 5,00 |
  | Latencia mediana | 9,9 s | 3,0 s |
  | Latencia media | 9,5 s | 9,9 s |
  | Tokens de razonamiento medios | 17 | 0 |

- **Decisión:** `thinking_level = "MINIMAL"` por defecto (`src/config.py`).
- **Por qué:** la calidad es igual o algo mejor (con 36 preguntas, las
  diferencias pequeñas pueden ser variación entre ejecuciones) y la latencia
  típica baja a un tercio. La lentitud de LOW no viene de razonar mucho
  (17 tokens de media), sino de activar el modo de razonamiento.
- **Lectura de la latencia:** se usa la mediana. La media de MINIMAL la
  suben 6 preguntas de unos 45 s, todas esperando el primer token: casi
  seguro esperas por el límite de 15 peticiones/min, que MINIMAL alcanza
  antes porque responde más deprisa. No es un coste del modo de razonamiento.
- **Consecuencias:** las preguntas que exigieran encadenar razonamientos
  podrían salir peor; en este dominio (redactar a partir de fragmentos
  recuperados y datos de herramientas) no se observa. LOW sigue disponible
  con `--thinking LOW` o cambiando la configuración.

## D13 · App web: Gradio como capa fina, un agente por visitante

- **Contexto:** el apartado opcional pide una interfaz web desplegada y
  observabilidad; la app usa la clave del autor como secreto (D11), con la
  cuota gratuita compartida entre todos los visitantes.
- **Decisiones:**
  - **Capa fina sobre el agente:** `app.py` solo conecta `chat_stream` (el
    streaming) y `last_trace` (la observabilidad) con Gradio. La
    presentación de la traza está en `src/trace_view.py`, sin Gradio, y se
    prueba con pytest.
  - **Un agente por visitante** (`gr.State`): con uno global, las memorias
    se mezclarían (el "allí" de un visitante se resolvería con el sitio de
    otro) y "Nueva conversación" borraría la de los demás. El índice de la
    guía, que no guarda estado de la conversación, se comparte y se
    construye una sola vez al arrancar.
  - **Límites** (`AppConfig`): 20 mensajes por conversación, 500 caracteres
    por mensaje y 2 peticiones a la vez. Una pregunta gasta ~3 peticiones
    de las 500 diarias.
  - **Modelos de respaldo activados** (en la evaluación se desactivaron
    para comparar en igualdad): si Flash-Lite agota su cuota, responde
    Gemini 3.6 Flash, que tiene la suya.
  - **Fuentes separadas por uso:** un fragmento cuenta como citado si
    alguna de sus páginas aparece citada en la respuesta. Los citados se
    muestran con su texto; los recuperados que no se usaron (p. ej. otros
    sitios de la misma zona que superan el umbral), solo con su título. Así
    se ve qué encontró la búsqueda sin confundirlo con la base de la
    respuesta.
  - **Texto palabra a palabra:** Flash-Lite envía el streaming en trozos
    grandes y la respuesta aparecía casi de golpe; la app los reparte en
    palabras con una pausa de 30 ms. Es solo presentación: la latencia del
    panel es la real.
  - **Despliegue en Render** (*Web Service* gratuito, desde el repositorio
    de GitHub) con dependencias reducidas (`deploy/requirements.txt`, sin
    Jupyter ni pytest). Se intentó primero Hugging Face Spaces, pero alojar
    Gradio en su CPU gratuita exige ya la suscripción PRO (error 402).
    Render, además, no expone los ficheros del repositorio: la guía no
    queda descargable.
  - **Acceso abierto, con inicio de sesión opcional** (`auth` de Gradio,
    leído de `APP_USER` y `APP_PASSWORD`): la demo se publica abierta para
    que se pueda probar sin credenciales; el riesgo es solo de cuota (capa
    gratuita, sin coste) y lo acotan los límites por conversación. Si hiciera
    falta restringirla, basta con definir esas dos variables en Render, sin
    tocar el código.
- **Alternativas:** Streamlit (reejecuta todo el script en cada interacción
  y el streaming de chat es menos directo); `gr.ChatInterface` (más corto,
  pero sin sitio para el panel de observabilidad).
