"""Registro de las llamadas a herramientas en formato JSON Lines.

Cada intento (válido o fallido) se escribe como una línea JSON independiente
en ``logs/tools.jsonl``. El formato permite añadir líneas sin reescribir el
archivo y leerlo después con pandas para la evaluación y el panel de
observabilidad.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import TOOL_LOG_PATH

LOG_PATH: Path = TOOL_LOG_PATH


def log_event(tool: str, **fields: Any) -> dict[str, Any]:
    """Añade una línea al registro de herramientas.

    Args:
        tool: Nombre de la herramienta, p. ej. ``"get_weather"``.
        **fields: Datos del intento (argumentos, resultado, error, latencia...).

    Returns:
        El registro escrito, con la marca de tiempo y la herramienta.
    """
    record = {"ts": datetime.now().isoformat(timespec="seconds"), "tool": tool}
    record.update(fields)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Lee todas las líneas del registro.

    Args:
        path: Archivo a leer; por defecto, el registro activo.

    Returns:
        Los registros en orden de escritura (lista vacía si no hay archivo).
    """
    path = path or LOG_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
