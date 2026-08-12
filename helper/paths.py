"""Path helpers for the DexGrasp-Zero repository."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Return a path relative to the repository's resource directory."""

    return PROJECT_ROOT.joinpath("rsc", *parts)
