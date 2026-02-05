#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys


def main() -> None:
    checks = {
        "python": sys.version.split()[0],
        "tesseract": bool(shutil.which("tesseract")),
        "poppler_pdftoppm": bool(shutil.which("pdftoppm")),
    }
    try:
        import pypdf  # type: ignore

        checks["pypdf"] = True
    except Exception:
        checks["pypdf"] = False
    try:
        import pdf2image  # type: ignore

        checks["pdf2image"] = True
    except Exception:
        checks["pdf2image"] = False
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

