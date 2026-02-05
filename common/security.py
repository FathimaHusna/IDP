from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import tempfile


def sniff_mime(data: bytes, filename: Optional[str] = None) -> str:
    # Minimal sniff: PDF header or fallback to extension
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".txt":
            return "text/plain"
    return "application/octet-stream"


def file_size_mb(raw: bytes) -> float:
    return len(raw) / (1024 * 1024)


def pdf_page_count_from_bytes(raw: bytes) -> Optional[int]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(raw))
        return len(reader.pages)
    except Exception:
        return None


def pdf_page_count_from_path(path: str) -> Optional[int]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(path)
        return len(reader.pages)
    except Exception:
        return None


def antivirus_scan_path(path: str) -> Tuple[bool, Optional[str]]:
    # Best-effort ClamAV hook; returns (clean, details)
    try:
        if shutil.which("clamscan") is None:
            return True, None
        import subprocess

        r = subprocess.run(["clamscan", "--no-summary", path], capture_output=True, text=True, timeout=30)
        out = r.stdout.strip() + r.stderr.strip()
        if "OK" in out and r.returncode in (0,):
            return True, None
        if r.returncode == 1 or "FOUND" in out:
            return False, out
        return True, None
    except Exception as e:
        # Fail-open (log only) to avoid blocking usage when AV not available
        return True, str(e)


def antivirus_scan_bytes(raw: bytes, suffix: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Write bytes to a temporary file and scan with ClamAV if available."""
    try:
        if shutil.which("clamscan") is None:
            return True, None
        ext = suffix or "bin"
        with tempfile.NamedTemporaryFile(prefix="upload_", suffix=f".{ext}", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            return antivirus_scan_path(tmp.name)
    except Exception as e:
        return True, str(e)
