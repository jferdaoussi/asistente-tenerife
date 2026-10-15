"""Índice vectorial de la guía en Chroma y búsqueda semántica.

Flujo:

1. **Indexación** (una vez): fragmentos -> embeddings -> colección de Chroma
   persistida en disco. Una huella (hash) de los fragmentos y del modelo de
   embeddings permite saber si el índice guardado sigue siendo válido; si
   algo cambia, se reconstruye automáticamente.
2. **Búsqueda** (cada pregunta): pregunta -> embedding -> los ``top_k``
   fragmentos más parecidos, con filtros opcionales por zona y categoría.
   Se descartan los que no superan la similitud mínima, para que el
   asistente pueda decir "eso no está en la guía" en lugar de inventar.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.chunking import Chunk, chunk_by_section, citation
from src.config import CHROMA_DIR, RAG, RAGConfig
from src.embeddings import GeminiEmbedder
from src.loader import load_pages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Fragmento recuperado para una pregunta.

    Attributes:
        text: Texto del fragmento, con su cabecera de contexto.
        sitio: Sitio de la guía al que pertenece.
        zona: Zona de la isla ("norte", "sur" o "").
        categoria: "lugar", "gastronomia" o "".
        page_start: Primera página del PDF.
        page_end: Última página del PDF.
        similarity: Similitud coseno con la pregunta (de -1 a 1).
    """

    text: str
    sitio: str
    zona: str
    categoria: str
    page_start: int
    page_end: int
    similarity: float

    @property
    def citation(self) -> str:
        """Cita legible del fragmento, p. ej. "TENERIFE.pdf, p. 23 · Masca"."""
        return citation(self.sitio, self.page_start, self.page_end)


def fingerprint(chunks: list[Chunk], config: RAGConfig) -> str:
    """Huella que identifica un índice: fragmentos y modelo de embeddings.

    Args:
        chunks: Fragmentos que se indexan.
        config: Parámetros del RAG.

    Returns:
        Hash SHA-256 en hexadecimal (16 caracteres).
    """
    digest = hashlib.sha256()
    digest.update(f"{config.embedding_model}|{config.embedding_dim}".encode())
    for chunk in chunks:
        digest.update(f"{chunk.id}|{chunk.text}|{chunk.metadata}".encode())
    return digest.hexdigest()[:16]


class GuideIndex:
    """Colección de Chroma con los fragmentos de la guía."""

    def __init__(
        self,
        embedder: GeminiEmbedder | None = None,
        config: RAGConfig = RAG,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str | None = None,
    ):
        """Abre (o crea) la base de datos vectorial en disco.

        Args:
            embedder: Generador de embeddings. Por defecto, Gemini.
            config: Parámetros del RAG.
            persist_dir: Carpeta donde Chroma guarda el índice.
            collection_name: Nombre de la colección; permite tener varias
                (p. ej. una por estrategia de chunking en la evaluación).
        """
        self.config = config
        self.embedder = embedder or GeminiEmbedder(config=config)
        self.collection_name = collection_name or config.collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))

    def _collection(self, metadata: dict | None = None) -> chromadb.Collection:
        """Devuelve la colección, creándola con distancia coseno si no existe."""
        return self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,  # los embeddings los calculamos nosotros
            configuration={"hnsw": {"space": "cosine"}},
            metadata=metadata,
        )

    @property
    def size(self) -> int:
        """Número de fragmentos indexados."""
        return self._collection().count()

    def build(self, chunks: list[Chunk], force: bool = False) -> bool:
        """Indexa los fragmentos si el índice guardado no está al día.

        Args:
            chunks: Fragmentos de la guía.
            force: Reconstruir aunque la huella coincida.

        Returns:
            True si se ha (re)construido el índice; False si se reutilizó.
        """
        expected = fingerprint(chunks, self.config)
        current = self._collection()
        if (
            not force
            and current.metadata
            and current.metadata.get("fingerprint") == expected
            and current.count() == len(chunks)
        ):
            logger.info("Índice '%s' al día, se reutiliza", self.collection_name)
            return False

        logger.info(
            "Indexando %d fragmentos en '%s'", len(chunks), self.collection_name
        )
        self._client.delete_collection(self.collection_name)
        collection = self._collection(metadata={"fingerprint": expected})
        vectors = self.embedder.embed_documents(
            [chunk.text for chunk in chunks], [chunk.sitio for chunk in chunks]
        )
        collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        return True

    def search(
        self,
        query: str,
        k: int | None = None,
        zona: str | None = None,
        categoria: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Busca los fragmentos más parecidos a una pregunta.

        Args:
            query: Pregunta del usuario.
            k: Número máximo de fragmentos. Por defecto, ``config.top_k``.
            zona: Restringe la búsqueda a "norte" o "sur".
            categoria: Restringe la búsqueda a "lugar" o "gastronomia".
            min_similarity: Similitud mínima. Por defecto,
                ``config.min_similarity``; 0 para no filtrar.

        Returns:
            Fragmentos ordenados de más a menos parecido.
        """
        k = k or self.config.top_k
        threshold = (
            self.config.min_similarity if min_similarity is None else min_similarity
        )
        filters = [
            {field: value}
            for field, value in (("zona", zona), ("categoria", categoria))
            if value
        ]
        where = (
            None
            if not filters
            else filters[0]
            if len(filters) == 1
            else {"$and": filters}
        )
        response = self._collection().query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results = [
            SearchResult(
                text=document,
                sitio=meta["sitio"],
                zona=meta["zona"],
                categoria=meta["categoria"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                # Chroma devuelve distancia coseno: distancia = 1 - similitud.
                similarity=1.0 - distance,
            )
            for document, meta, distance in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=True,
            )
        ]
        return [result for result in results if result.similarity >= threshold]


def format_context(results: list[SearchResult]) -> str:
    """Prepara los fragmentos recuperados para insertarlos en el prompt.

    Cada fragmento se numera y lleva su cita, para que el LLM pueda
    referenciar la fuente exacta en su respuesta.

    Args:
        results: Fragmentos devueltos por :meth:`GuideIndex.search`.

    Returns:
        Bloque de texto con los fragmentos, o un aviso si no hay ninguno.
    """
    if not results:
        return "No se ha encontrado información relevante en la guía."
    return "\n\n".join(
        f"[{n}] Fuente: {result.citation}\n{result.text}"
        for n, result in enumerate(results, start=1)
    )


def build_index(force: bool = False) -> GuideIndex:
    """Carga la guía, la trocea por sitio y devuelve el índice listo para usar.

    Args:
        force: Reconstruir el índice aunque ya exista y esté al día.

    Returns:
        Índice de la guía.
    """
    index = GuideIndex()
    index.build(chunk_by_section(load_pages()), force=force)
    return index
