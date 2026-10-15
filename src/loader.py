"""Carga y limpieza del texto de la guía turística en PDF.

El PDF se generó con "Imprimir a PDF" desde Word y no conserva los
hipervínculos: las marcas como ``[vídeo - ubicación]`` o ``[web]`` son texto
sin URL detrás. Se eliminan porque no aportan información y confundirían al
modelo de embeddings y al LLM.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.config import GUIDE_PATH

_LINK_WORDS = r"(?:vídeo|video|ubicación|web|tripadvisor)"

#: Patrones de las marcas de enlace que se eliminan del texto.
LINK_PATTERNS = (
    # [vídeo - ubicación], [ubicación], [web], [Tripadvisor]...
    re.compile(rf"\s*\[(?:\s*{_LINK_WORDS}\s*[-–]?)+\s*\]", re.IGNORECASE),
    # (ubicación)
    re.compile(rf"\s*\(\s*{_LINK_WORDS}\s*\)", re.IGNORECASE),
    # "Plaza Weyler - ubicación –" (enlace intercalado entre guiones)
    re.compile(r"\s*[-–]\s*ubicación\s*[-–]?", re.IGNORECASE),
)


@dataclass(frozen=True)
class Page:
    """Texto limpio de una página de la guía.

    Attributes:
        number: Número de página, empezando en 1 como en el visor de PDF.
        text: Texto de la página sin marcas de enlace.
    """

    number: int
    text: str


def clean_text(text: str) -> str:
    """Elimina marcas de enlace y espacios sobrantes de un texto.

    Args:
        text: Texto extraído de una página del PDF.

    Returns:
        Texto con una línea por línea original, sin espacios al final.
    """
    for pattern in LINK_PATTERNS:
        text = pattern.sub("", text)
    return "\n".join(line.strip() for line in text.splitlines())


def load_pages(path: Path = GUIDE_PATH) -> list[Page]:
    """Extrae y limpia el texto de cada página del PDF.

    Args:
        path: Ruta del PDF de la guía.

    Returns:
        Páginas en orden, incluidas las que solo contienen imágenes (con
        texto vacío), para que la numeración coincida con el PDF.

    Raises:
        FileNotFoundError: Si el PDF no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra la guía en {path}")
    reader = PdfReader(path)
    return [
        Page(number=i, text=clean_text(page.extract_text() or ""))
        for i, page in enumerate(reader.pages, start=1)
    ]
