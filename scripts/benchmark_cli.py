#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List

from agents.agent_orchestrator import run_folder_with_recon
from common.paths import LOGS_DIR, ensure_dirs


def _safe_mean(vals: List[float]) -> float:
    return round(mean(vals), 4) if vals else 0.0


def compute_metrics(results: List[Dict[str, Any]], chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    routes = {}
    types = {}
    ocr_q = []
    confs = []
    completeness = []
    val_scores = []
    reasons = {}
    errors = 0
    for r in results:
        if r.get("error"):
            errors += 1
            continue
        route = r.get("route") or "unknown"
        routes[route] = routes.get(route, 0) + 1
        dt = r.get("type") or "unknown"
        types[dt] = types.get(dt, 0) + 1
        if r.get("ocr_quality") is not None:
            ocr_q.append(float(r["ocr_quality"]))
        if r.get("final_confidence") is not None:
            confs.append(float(r["final_confidence"]))
        v = r.get("validation") or {}
        if v.get("completeness") is not None:
            completeness.append(float(v.get("completeness")))
        if v.get("validation_score") is not None:
            val_scores.append(float(v.get("validation_score")))
        for rc in (r.get("reason_codes") or []):
            reasons[rc] = reasons.get(rc, 0) + 1

    recon_status = {}
    anomalies = {}
    for c in chains:
        s = c.get("status") or "unknown"
        recon_status[s] = recon_status.get(s, 0) + 1
        for a in (c.get("anomalies") or []):
            anomalies[a] = anomalies.get(a, 0) + 1

    return {
        "counts": {
            "documents": len(results),
            "errors": errors,
        },
        "routes": routes,
        "types": types,
        "means": {
            "ocr_quality": _safe_mean(ocr_q),
            "final_confidence": _safe_mean(confs),
            "completeness": _safe_mean(completeness),
            "validation_score": _safe_mean(val_scores),
        },
        "reconciliation": {
            "status_counts": recon_status,
            "anomalies": anomalies,
        },
        "reason_codes": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark IDP pipeline on a folder and emit metrics.json")
    ap.add_argument("folder", help="Folder of files to process")
    ap.add_argument("--out", default=str(LOGS_DIR / "metrics.json"), help="Output metrics path")
    args = ap.parse_args()

    ensure_dirs()
    results, chains = run_folder_with_recon(args.folder)
    metrics = compute_metrics(results, chains)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metrics → {outp}")
    # Also write a timestamped copy for history dashboards
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    hist = outp.parent / f"metrics-{ts}.json"
    hist.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
