"""Tests del pipeline RAG: limpieza, chunking e índice (sin llamadas a la API)."""

import math
import re

import pytest

from src.chunking import (
    Chunk,
    chunk_by_section,
    chunk_fixed,
    chunk_recursive,
    citation,
)
from src.config import RAGConfig
from src.loader import clean_text, load_pages
from src.rag import GuideIndex, fingerprint, format_context


@pytest.fixture(scope="module")
def pages():
    return load_pages()


@pytest.fixture(scope="module")
def chunks(pages):
    return chunk_by_section(pages)


# --- Limpieza ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("o Plaza de España [vídeo - ubicación]", "o Plaza de España"),
        (
            "Terrazas del Sauzal (zona de El Sauzal [ubicación])",
            "Terrazas del Sauzal (zona de El Sauzal)",
        ),
        (
            "Aparcar en el Parque Marítimo (ubicación).",
            "Aparcar en el Parque Marítimo.",
        ),
        ("hacia Plaza Weyler - ubicación –; ir", "hacia Plaza Weyler; ir"),
        ("Guachinche El Cubano [Tripadvisor]", "Guachinche El Cubano"),
    ],
)
def test_clean_text_removes_dead_link_labels(raw, expected):
    assert clean_text(raw) == expected


def test_pages_keep_pdf_numbering(pages):
    assert len(pages) == 25
    assert [p.number for p in pages] == list(range(1, 26))
    assert not re.search(r"\[(vídeo|ubicación|web)", " ".join(p.text for p in pages))


# --- Chunking por sitio -----------------------------------------------------


def test_every_chunk_has_context_header(chunks):
    for chunk in chunks:
        header = chunk.text.splitlines()[0]
        assert " › " in header
        assert chunk.sitio in header


def test_chunks_respect_max_size(chunks):
    assert all(len(chunk.text) <= RAGConfig().chunk_max_chars for chunk in chunks)


def test_chunk_ids_are_unique(chunks):
    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_zones_and_categories(chunks):
    by_site = {chunk.sitio: chunk for chunk in chunks}
    assert by_site["La Orotava"].zona == "norte"
    assert by_site["Costa Adeje"].zona == "sur"
    assert by_site["Masca"].zona == "sur"
    assert by_site["Sitios para comer"].categoria == "gastronomia"
    assert by_site["Sitios para comer"].zona == "norte"


def test_page_ranges_are_valid(chunks):
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 25


def test_known_fact_lands_in_right_site(chunks):
    teno = [c for c in chunks if "línea 369" in c.text]
    assert len(teno) == 1
    assert teno[0].sitio == "Isla Baja"


# --- Chunking de tamaño fijo ------------------------------------------------


def test_fixed_chunks_have_bounded_size_and_overlap(pages):
    fixed = chunk_fixed(pages, size=500, overlap=100)
    assert all(len(chunk.text) <= 500 for chunk in fixed)
    # El final de un fragmento reaparece al principio del siguiente.
    first, second = fixed[0].text, fixed[1].text
    assert second.split()[0] in first.split()[-25:]


def test_fixed_chunks_reject_invalid_overlap(pages):
    with pytest.raises(ValueError):
        chunk_fixed(pages, size=100, overlap=100)


# --- Chunking de LangChain (referencia) -------------------------------------


def test_langchain_chunks_are_bounded_and_mapped_to_pages(pages):
    chunks = chunk_recursive(pages, size=800, overlap=150)
    assert all(len(chunk.text) <= 800 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 25
        # La página asignada contiene de verdad el inicio del fragmento.
        start = chunk.text.split("\n")[0][:30]
        assert start in pages[chunk.page_start - 1].text


# --- Citas, contexto y huella -----------------------------------------------


def test_citation_format():
    assert citation("Masca", 22, 22) == "TENERIFE.pdf, p. 22 · Masca"
    assert citation("La Orotava", 8, 9) == "TENERIFE.pdf, p. 8-9 · La Orotava"
    assert citation("", 3, 3) == "TENERIFE.pdf, p. 3"


def test_format_context_without_results():
    assert "No se ha encontrado" in format_context([])


def test_fingerprint_changes_with_content(chunks):
    config = RAGConfig()
    assert fingerprint(chunks, config) == fingerprint(list(chunks), config)
    assert fingerprint(chunks, config) != fingerprint(chunks[:-1], config)
    other_model = RAGConfig(embedding_model="otro-modelo")
    assert fingerprint(chunks, config) != fingerprint(chunks, other_model)


# --- Índice con un embedder falso (sin API) ---------------------------------


class FakeEmbedder:
    """Bolsa de palabras sobre un vocabulario fijo: determinista y sin red."""

    VOCAB = ["playa", "teide", "comer", "fiesta", "barranco", "norte", "sur"]

    def _vector(self, text):
        words = text.lower()
        values = [float(words.count(term)) + 0.01 for term in self.VOCAB]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def embed_documents(self, texts, titles):
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)


def _chunk(chunk_id, text, zona):
    return Chunk(chunk_id, text, chunk_id, zona, "lugar", 1, 1)


@pytest.fixture()
def index(tmp_path):
    index = GuideIndex(embedder=FakeEmbedder(), persist_dir=tmp_path)
    index.build(
        [
            _chunk("benijo", "Playa de arena negra en el norte", "norte"),
            _chunk("teide", "Subir al Teide y ver el Teide nevado", "norte"),
            _chunk("adeje", "Fiesta y playa en el sur", "sur"),
        ]
    )
    return index


def test_index_is_reused_when_unchanged(index):
    chunks = [
        _chunk("benijo", "Playa de arena negra en el norte", "norte"),
        _chunk("teide", "Subir al Teide y ver el Teide nevado", "norte"),
        _chunk("adeje", "Fiesta y playa en el sur", "sur"),
    ]
    assert index.build(chunks) is False
    assert index.build(chunks, force=True) is True
    assert index.size == 3


def test_search_ranks_by_similarity(index):
    results = index.search("¿cómo subir al teide?", k=3, min_similarity=0)
    assert results[0].sitio == "teide"
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_search_filters_by_zone(index):
    results = index.search("playa", k=3, zona="sur", min_similarity=0)
    assert [r.sitio for r in results] == ["adeje"]


def test_search_applies_threshold(index):
    results = index.search("teide", k=3, min_similarity=0.9)
    assert [r.sitio for r in results] == ["teide"]
