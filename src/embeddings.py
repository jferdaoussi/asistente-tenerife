"""Generación de embeddings con Gemini Embedding 2.

Gemini Embedding 2 no admite el parámetro ``task_type``: el uso se indica
con prefijos de texto. Los fragmentos y las preguntas se formatean de forma
distinta (embedding asimétrico) porque una pregunta corta y un párrafo largo
no se parecen en la forma aunque hablen de lo mismo.
"""

from google.genai import types

from src.config import RAG, RAGConfig
from src.llm import REQUEST_ERRORS, GeminiClient, LLMError, friendly_message

DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
QUERY_TEMPLATE = "task: search result | query: {query}"


class GeminiEmbedder:
    """Convierte fragmentos y preguntas en vectores normalizados."""

    def __init__(self, client: GeminiClient | None = None, config: RAGConfig = RAG):
        """Crea el generador de embeddings.

        Args:
            client: Cliente de Gemini; se reutilizan su clave y sus reintentos.
            config: Parámetros del RAG (modelo, dimensión, tamaño de lote).
        """
        self.config = config
        self._sdk = (client or GeminiClient()).sdk

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Pide a la API un vector por cada texto, por lotes.

        Args:
            texts: Textos ya formateados con su prefijo.

        Returns:
            Un vector por texto, en el mismo orden.

        Raises:
            LLMError: Si la API falla tras los reintentos del SDK.
        """
        vectors: list[list[float]] = []
        size = self.config.embed_batch_size
        for start in range(0, len(texts), size):
            # Cada texto va en su propio Content: si se pasaran como partes
            # de uno solo, la API devolvería un único vector agregado.
            batch = [
                types.Content(parts=[types.Part(text=text)])
                for text in texts[start : start + size]
            ]
            try:
                response = self._sdk.models.embed_content(
                    model=self.config.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.config.embedding_dim
                    ),
                )
            except REQUEST_ERRORS as err:
                raise LLMError(friendly_message(err)) from err
            vectors.extend(embedding.values for embedding in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]:
        """Calcula los embeddings de los fragmentos de la guía.

        Args:
            texts: Texto de cada fragmento.
            titles: Título de cada fragmento (el sitio), o "" si no tiene.

        Returns:
            Un vector por fragmento.
        """
        formatted = [
            DOCUMENT_TEMPLATE.format(title=title or "none", text=text)
            for text, title in zip(texts, titles, strict=True)
        ]
        return self._embed(formatted)

    def embed_query(self, query: str) -> list[float]:
        """Calcula el embedding de una pregunta del usuario.

        Args:
            query: Pregunta en lenguaje natural.

        Returns:
            Vector de la pregunta.
        """
        return self._embed([QUERY_TEMPLATE.format(query=query)])[0]
