"""Publica la app en un Space de Hugging Face.

Sube solo lo que necesita la app: ``app.py``, el código de ``src/``, la guía
y los ficheros de esta carpeta (ficha del Space y dependencias reducidas).
La clave de Gemini no se sube: se configura como secreto ``GEMINI_API_KEY``
en *Settings → Variables and secrets* del Space.

Uso (tras ``hf auth login``)::

    python deploy/publish.py [--repo usuario/asistente-tenerife]
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


def main() -> None:
    """Crea el Space si no existe y sube los ficheros de la app."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="Space de destino (usuario/nombre)")
    args = parser.parse_args()

    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/asistente-tenerife"
    api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)

    files = {
        DEPLOY / "README.md": "README.md",
        DEPLOY / "requirements.txt": "requirements.txt",
        ROOT / "app.py": "app.py",
        ROOT / "data" / "TENERIFE.pdf": "data/TENERIFE.pdf",
    }
    files |= {path: f"src/{path.name}" for path in (ROOT / "src").glob("*.py")}
    api.create_commit(
        repo_id=repo,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
            for local, remote in files.items()
        ],
        commit_message="Publica la app",
    )
    print(f"Publicado en https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
