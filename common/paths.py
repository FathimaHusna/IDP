from __future__ import annotations

from pathlib import Path

# Centralized project paths
ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
GRAPHS_DIR = ROOT / "graphs"


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)

