from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .paths import LOGS_DIR


class Storage:
    def __init__(self) -> None:
        self.backend = os.environ.get("STORAGE_BACKEND", "local")

    def save_bytes(self, rel_path: str, data: bytes) -> str:
        if self.backend == "local":
            p = LOGS_DIR / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return str(p)
        # Placeholder for S3/GCS: fail back to local
        p = LOGS_DIR / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)
