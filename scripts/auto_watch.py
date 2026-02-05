#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from agents.agent_orchestrator import run_on_path
from common.paths import LOGS_DIR, ensure_dirs


ALLOWED_SUFFIX = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".json"}


def compute_metrics(results: List[Dict[str, Any]], chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    routes: Dict[str, int] = {}
    types: Dict[str, int] = {}
    ocr_q: List[float] = []
    confs: List[float] = []
    completeness: List[float] = []
    val_scores: List[float] = []
    reasons: Dict[str, int] = {}
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

    recon_status: Dict[str, int] = {}
    anomalies: Dict[str, int] = {}
    for c in chains:
        s = c.get("status") or "unknown"
        recon_status[s] = recon_status.get(s, 0) + 1
        for a in (c.get("anomalies") or []):
            anomalies[a] = anomalies.get(a, 0) + 1

    from statistics import mean

    def _m(v: List[float]) -> float:
        return round(mean(v), 4) if v else 0.0

    return {
        "counts": {"documents": len(results), "errors": errors},
        "routes": routes,
        "types": types,
        "means": {
            "ocr_quality": _m(ocr_q),
            "final_confidence": _m(confs),
            "completeness": _m(completeness),
            "validation_score": _m(val_scores),
        },
        "reconciliation": {"status_counts": recon_status, "anomalies": anomalies},
        "reason_codes": reasons,
    }


def scan_files(folder: Path, seen: Set[Tuple[str, float]]) -> List[Path]:
    found: List[Path] = []
    for p in sorted(folder.glob("**/*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX:
            key = (str(p), p.stat().st_mtime)
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a folder and auto-run the agentic pipeline on new files")
    ap.add_argument("folder", help="Folder to watch")
    ap.add_argument("--interval", type=float, default=10.0, help="Scan interval seconds")
    ap.add_argument("--audit", action="store_true", help="Enable audit (IDP_AUDIT=1)")
    args = ap.parse_args()

    ensure_dirs()
    watch = Path(args.folder)
    if not watch.exists():
        print(f"Watch folder not found: {watch}")
        sys.exit(1)

    # Best-effort enable audit if requested
    if args.audit:
        import os

        os.environ["IDP_AUDIT"] = "1"

    seen: Set[Tuple[str, float]] = set()
    results: List[Dict[str, Any]] = []
    chains: List[Dict[str, Any]] = []
    run_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_path = LOGS_DIR / "runs" / f"watch-{run_ts}.jsonl"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Watching {watch} every {args.interval}s. Writing runs to {run_path}")

    stop = False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    while not stop:
        new_files = scan_files(watch, seen)
        if new_files:
            for fp in new_files:
                try:
                    r = run_on_path(str(fp))
                    results.append(r)
                    with open(run_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                except Exception as e:
                    results.append({"file": str(fp), "error": str(e)})
            # Write metrics snapshot after each batch
            m = compute_metrics(results, chains)
            (LOGS_DIR / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
            hist = LOGS_DIR / f"metrics-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
            hist.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Processed {len(new_files)} file(s). Metrics updated.")
        time.sleep(args.interval)

    print("Exiting watcher.")


if __name__ == "__main__":
    main()

