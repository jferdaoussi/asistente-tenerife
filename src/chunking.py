"""División de la guía en fragmentos (chunks) para la búsqueda semántica.

Se ofrecen tres estrategias:

- :func:`chunk_by_section` (la que usa el asistente): aprovecha la estructura
  de la guía. Cada fragmento pertenece a un único sitio (La Orotava, Masca...)
  y lleva una cabecera de contexto ("Zona Norte › La Orotava") para que
  ningún trozo quede huérfano. Las secciones largas se dividen por sus puntos
  de interés, nunca a mitad de frase.
- :func:`chunk_fixed`: ventanas de tamaño fijo con solapamiento, sin tener
  en cuenta la estructura. Sirve de referencia en la evaluación.
- :func:`chunk_recursive`: el ``RecursiveCharacterTextSplitter`` de
  LangChain, la referencia del sector. También se usa solo para comparar.

Las dos primeras son funciones propias: ninguna librería genérica conoce las
marcas de esta guía ("•" para sitios, "o" para puntos de interés, "▪" para
restaurantes). :func:`chunk_by_section` aplica la misma idea que el
``chunk_by_title`` de Unstructured adaptada a ellas, y :func:`chunk_fixed`
equivale al ``CharacterTextSplitter`` de LangChain. Justificación completa en
``docs/decisiones.md`` (D4).
"""

import re
import unicodedata
from dataclasses import dataclass

from src.config import RAG
from src.loader import Page

SOURCE_NAME = "TENERIFE.pdf"

#: Cabeceras de zona de la guía y la zona que representan.
ZONE_HEADERS = {
    "ZONA NORTE": "norte",
    "ZONA SUR": "sur",
    "OTRAS VAINAS": "",
}
ZONE_LABELS = {"norte": "Zona Norte", "sur": "Zona Sur", "": "Tenerife"}

#: Líneas de la portada que no aportan contenido.
SKIP_LINES = {"TENERIFE – LUGARES DE INTERÉS", "SITIOS QUE VER"}

SECTION_MARK = "• "
#: Marcas que inician un nuevo bloque dentro de una sección.
BLOCK_START = re.compile(r"^(?:o |▪ |- |Nota:|Curiosidades:)")
#: Subcabeceras de zona dentro de "Sitios para comer" (p. ej. "o Zona Norte").
SUBZONE = re.compile(r"^o Zona (Norte|Sur)$")
#: Cita de página en un texto: "p. 8" o "p. 8-11".
PAGE_CITATION = re.compile(r"p\.\s*(\d+)(?:\s*[-–]\s*(\d+))?")


@dataclass(frozen=True)
class Chunk:
    """Fragmento de la guía listo para indexar.

    Attributes:
        id: Identificador único y estable del fragmento.
        text: Texto con la cabecera de contexto incluida.
        sitio: Sitio de la guía al que pertenece ("" si no se conoce).
        zona: "norte", "sur" o "" si no se conoce.
        categoria: "lugar", "gastronomia" o "" si no se conoce.
        page_start: Primera página del PDF que abarca.
        page_end: Última página del PDF que abarca.
    """

    id: str
    text: str
    sitio: str
    zona: str
    categoria: str
    page_start: int
    page_end: int

    @property
    def metadata(self) -> dict[str, str | int]:
        """Metadatos para Chroma (solo admite valores escalares)."""
        return {
            "sitio": self.sitio,
            "zona": self.zona,
            "categoria": self.categoria,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass
class _Block:
    """Párrafo o punto de interés de una sección, con sus páginas."""

    text: str
    page_start: int
    page_end: int


@dataclass
class _Section:
    """Sección de la guía dedicada a un sitio.

    ``title`` es el título completo, que va en la cabecera del fragmento
    porque el paréntesis suele aportar contexto útil para la búsqueda.
    """

    title: str
    zona: str
    categoria: str
    blocks: list[_Block]


def _parse_sections(pages: list[Page]) -> list[_Section]:
    """Recorre la guía línea a línea y la agrupa en secciones y bloques.

    Args:
        pages: Páginas limpias de la guía.

    Returns:
        Secciones en el orden de la guía.
    """
    sections: list[_Section] = []
    zona, categoria = "", "lugar"
    new_block = True
    for page in pages:
        for raw in page.text.splitlines():
            line = raw.strip()
            if not line:
                new_block = True
                continue
            if line in SKIP_LINES:
                continue
            if line in ZONE_HEADERS:
                zona = ZONE_HEADERS[line]
                categoria = "lugar" if zona else "gastronomia"
                continue
            if line.startswith(SECTION_MARK):
                title = line.removeprefix(SECTION_MARK).rstrip(":").strip()
                sections.append(_Section(title, zona, categoria, []))
                new_block = True
                continue
            if match := SUBZONE.match(line):
                zona = match.group(1).lower()
                sections[-1].zona = zona
                continue
            if not sections:
                continue
            blocks = sections[-1].blocks
            if new_block or BLOCK_START.match(line) or not blocks:
                blocks.append(_Block(line, page.number, page.number))
            else:
                blocks[-1].text += " " + line
                blocks[-1].page_end = page.number
            new_block = False
    return sections


def _header(zona: str, sitio: str) -> str:
    """Cabecera de contexto de un fragmento, p. ej. "Zona Norte › Masca"."""
    return f"{ZONE_LABELS[zona]} › {sitio}"


def _slug(text: str) -> str:
    """Convierte un nombre en un identificador apto para Chroma."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.decode().lower()).strip("-")[:40]


def _short_name(title: str) -> str:
    """Nombre corto de un sitio para las citas, sin el paréntesis final.

    Por ejemplo, "Isla Baja (Icod de los Vinos, ...)" pasa a "Isla Baja".
    """
    return re.sub(r"\s*\(.*\)$", "", title)


def chunk_by_section(
    pages: list[Page], max_chars: int = RAG.chunk_max_chars
) -> list[Chunk]:
    """Divide la guía en fragmentos que respetan sitios y puntos de interés.

    Cada sección se convierte en un único fragmento si cabe en ``max_chars``;
    si no, sus bloques se agrupan de forma consecutiva en varios fragmentos,
    todos con la misma cabecera de contexto.

    Args:
        pages: Páginas limpias de la guía.
        max_chars: Tamaño máximo orientativo de cada fragmento. Un bloque que
            por sí solo lo supere se mantiene entero.

    Returns:
        Fragmentos en el orden de la guía.
    """
    chunks: list[Chunk] = []
    for section in _parse_sections(pages):
        header = _header(section.zona, section.title)
        sitio = _short_name(section.title)
        groups: list[list[_Block]] = [[]]
        size = len(header)
        for block in section.blocks:
            if groups[-1] and size + len(block.text) + 1 > max_chars:
                groups.append([])
                size = len(header)
            groups[-1].append(block)
            size += len(block.text) + 1
        for n, group in enumerate(g for g in groups if g):
            body = "\n".join(block.text for block in group)
            chunks.append(
                Chunk(
                    id=f"{_slug(sitio)}-{n}",
                    text=f"{header}\n{body}",
                    sitio=sitio,
                    zona=section.zona,
                    categoria=section.categoria,
                    page_start=group[0].page_start,
                    page_end=group[-1].page_end,
                )
            )
    return chunks


def chunk_fixed(
    pages: list[Page],
    size: int = RAG.fixed_chunk_size,
    overlap: int = RAG.fixed_chunk_overlap,
) -> list[Chunk]:
    """Divide la guía en ventanas de tamaño fijo con solapamiento.

    Ignora la estructura de la guía: los cortes pueden caer a mitad de un
    sitio, y los fragmentos no tienen cabecera ni metadatos de sitio o zona.
    Los cortes se ajustan al espacio más cercano para no partir palabras.

    Args:
        pages: Páginas limpias de la guía.
        size: Longitud de cada ventana, en caracteres.
        overlap: Caracteres compartidos entre ventanas consecutivas.

    Returns:
        Fragmentos en el orden de la guía.

    Raises:
        ValueError: Si el solapamiento no es menor que el tamaño.
    """
    if not 0 <= overlap < size:
        raise ValueError("El solapamiento debe ser menor que el tamaño")
    text, page_at = "", []
    for page in pages:
        content = " ".join(page.text.split())
        if content:
            text += content + " "
            page_at += [page.number] * (len(content) + 1)

    chunks: list[Chunk] = []
    start = 0
    while start < len(text.rstrip()):
        end = min(start + size, len(text))
        if end < len(text):
            end = text.rfind(" ", start + size // 2, end) + 1 or end
        piece = text[start:end].strip()
        chunks.append(
            Chunk(
                id=f"fijo-{len(chunks)}",
                text=piece,
                sitio="",
                zona="",
                categoria="",
                page_start=page_at[start],
                page_end=page_at[end - 1],
            )
        )
        if end >= len(text):
            break
        next_start = text.find(" ", end - overlap) + 1
        start = next_start if next_start > start else end
    return chunks


def chunk_recursive(
    pages: list[Page],
    size: int = RAG.fixed_chunk_size,
    overlap: int = RAG.fixed_chunk_overlap,
) -> list[Chunk]:
    """Divide la guía con el ``RecursiveCharacterTextSplitter`` de LangChain.

    El splitter corta primero por párrafos y, si un trozo sigue siendo mayor
    que ``size``, por líneas, luego por espacios y por último por caracteres.
    Respeta el texto mejor que :func:`chunk_fixed`, pero no conoce la
    estructura de la guía: no hay cabeceras ni metadatos de sitio o zona.

    Se importa dentro de la función porque LangChain solo hace falta para
    la evaluación, no para usar el asistente.

    Args:
        pages: Páginas limpias de la guía.
        size: Tamaño máximo de cada fragmento, en caracteres.
        overlap: Caracteres compartidos entre fragmentos consecutivos.

    Returns:
        Fragmentos en el orden de la guía.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text, page_at = "", []
    for page in pages:
        if page.text.strip():
            content = page.text.strip() + "\n\n"
            text += content
            page_at += [page.number] * len(content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    return [
        Chunk(
            id=f"langchain-{n}",
            text=doc.page_content,
            sitio="",
            zona="",
            categoria="",
            page_start=page_at[doc.metadata["start_index"]],
            page_end=page_at[doc.metadata["start_index"] + len(doc.page_content) - 1],
        )
        for n, doc in enumerate(splitter.create_documents([text]))
    ]


def citation(sitio: str, page_start: int, page_end: int) -> str:
    """Texto de cita de un fragmento, p. ej. "TENERIFE.pdf, p. 8-9 · La Orotava".

    Args:
        sitio: Sitio de la guía ("" si no se conoce).
        page_start: Primera página.
        page_end: Última página.

    Returns:
        Cita legible para mostrar al usuario.
    """
    pages = (
        f"p. {page_start}"
        if page_start == page_end
        else (f"p. {page_start}-{page_end}")
    )
    suffix = f" · {sitio}" if sitio else ""
    return f"{SOURCE_NAME}, {pages}{suffix}"


def cited_pages(text: str) -> set[int]:
    """Páginas citadas en un texto ("p. 8" o "p. 8-11"), con los rangos expandidos.

    Sirve para una respuesta del asistente y para la cita de un fragmento.
    """
    pages: set[int] = set()
    for start, end in PAGE_CITATION.findall(text):
        first = int(start)
        pages.update(range(first, int(end or first) + 1))
    return pages
