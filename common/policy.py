from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml  # type: ignore

DEFAULT_POLICY: Dict[str, Any] = {
    "ocr": {"min_quality_auto_accept": 0.75, "min_quality_process": 0.30},
    "routing": {
        "auto_accept_min_confidence": {"default": 0.90, "receipt": 0.90},
        "review_min_confidence": {"default": 0.70, "receipt": 0.60},
    },
    "anomalies": {"force_review": ["totals_mismatch", "missing_total", "missing_invoice_number"]},
    "vendors": {"allow": [], "deny": []},
    "limits": {"max_doc_size_mb": 20, "max_pages_ocr": 3},
}


def load_policy(path: str | Path = Path("config/policy.yaml")) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return DEFAULT_POLICY
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
            # shallow merge
            pol = DEFAULT_POLICY.copy()
            pol.update(obj)
            return pol
    except Exception:
        return DEFAULT_POLICY

