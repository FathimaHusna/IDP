#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore


def load_metrics(p: str | Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_gates(p: str | Path) -> Dict[str, Any]:
    obj = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    return obj.get("gates", {})


def check(metrics: Dict[str, Any], gates: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = metrics.get("counts", {})
    types = metrics.get("types", {})
    means = metrics.get("means", {})
    reasons = metrics.get("reason_codes", {})

    # Basic counts
    if counts.get("documents", 0) < gates.get("min_documents", 1):
        errors.append(f"min_documents failed: {counts.get('documents')} < {gates.get('min_documents')}")
    if counts.get("errors", 0) > gates.get("max_errors", 0):
        errors.append(f"max_errors failed: {counts.get('errors')} > {gates.get('max_errors')}")

    # Means
    if means.get("ocr_quality", 1.0) < gates.get("min_mean_ocr_quality", 0.0):
        errors.append("min_mean_ocr_quality failed")
    if means.get("final_confidence", 1.0) < gates.get("min_mean_final_confidence", 0.0):
        errors.append("min_mean_final_confidence failed")

    # Expected types minimums
    exp_types = gates.get("expected_types", {})
    for t, constraints in exp_types.items():
        min_required = constraints.get("min")
        if isinstance(min_required, (int, float)):
            if (types.get(t, 0) or 0) < min_required:
                errors.append(f"expected_types {t} min failed: {types.get(t, 0)} < {min_required}")

    # Forbid reason codes
    forbids = gates.get("forbid_reason_codes", []) or []
    for rc in forbids:
        if reasons.get(rc, 0) > 0:
            errors.append(f"forbidden reason code {rc} present")

    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Check acceptance gates against metrics and exit non-zero on failure")
    ap.add_argument("--metrics", default="logs/metrics.json")
    ap.add_argument("--gates", default="config/acceptance.yaml")
    args = ap.parse_args()

    m = load_metrics(args.metrics)
    g = load_gates(args.gates)
    errs = check(m, g)
    if errs:
        print("ACCEPTANCE FAILED:")
        for e in errs:
            print("- ", e)
        sys.exit(1)
    print("Acceptance gates passed.")


if __name__ == "__main__":
    main()
