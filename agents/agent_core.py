from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import time
from pathlib import Path
from uuid import uuid4


@dataclass
class Blackboard:
    data: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def log(self, msg: str) -> None:
        self.logs.append(msg)


class Agent:
    name: str = "agent"

    def run(self, bb: Blackboard) -> None:
        raise NotImplementedError


class Orchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def execute(self, bb: Optional[Blackboard] = None) -> Blackboard:
        bb = bb or Blackboard()
        # Initialize run id if requested
        if bb.get("run_id") is None and os.environ.get("IDP_AUDIT", "0") == "1":
            bb.set("run_id", str(uuid4()))
        # Time budgets
        try:
            agent_budget_ms = int(os.environ.get("IDP_AGENT_TIMEOUT_MS", "4000"))
        except Exception:
            agent_budget_ms = 4000
        try:
            global_sla_ms = int(os.environ.get("IDP_GLOBAL_SLA_MS", "12000"))
        except Exception:
            global_sla_ms = 12000
        t_start = time.perf_counter()
        for a in self.agents:
            try:
                t0 = time.perf_counter()
                bb.log(f"→ {a.name}: start")
                a.run(bb)
                dt = time.perf_counter() - t0
                ev = {"agent": a.name, "status": "ok", "duration_ms": int(dt * 1000)}
                bb.events.append(ev)
                bb.log(f"✓ {a.name}: {dt:.3f}s")
                # Soft per-agent timeout
                if ev["duration_ms"] > agent_budget_ms:
                    bb.log(f"⏱ agent timeout: {a.name} exceeded {agent_budget_ms} ms")
                    reasons = set(bb.get("reason_codes") or [])
                    reasons.add(f"timeout:{a.name}")
                    bb.set("reason_codes", sorted(reasons))
                    bb.set("route", "review")
            except Exception as e:
                dt = time.perf_counter() - t0
                bb.log(f"✗ {a.name}: error: {e}")
                ev = {"agent": a.name, "status": "error", "error": str(e), "duration_ms": int(dt * 1000)}
                bb.events.append(ev)
                # Minimal self-healing: mark for review and continue
                bb.set("route", "review")
            # Append to audit file per agent if enabled
            if os.environ.get("IDP_AUDIT", "0") == "1":
                run_id = bb.get("run_id")
                if run_id:
                    try:
                        from common.paths import LOGS_DIR
                        from common.db import insert_agent_event

                        (LOGS_DIR / "runs").mkdir(parents=True, exist_ok=True)
                        p = LOGS_DIR / "runs" / f"{run_id}.jsonl"
                        with open(p, "a", encoding="utf-8") as f:
                            import json

                            payload = {"run_id": run_id, **bb.events[-1]}
                            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        # Also persist to sqlite
                        insert_agent_event(run_id, bb.events[-1])
                    except Exception:
                        pass
            # Global SLA soft-cancel
            if (time.perf_counter() - t_start) * 1000 > global_sla_ms:
                bb.log(f"⏹ global SLA exceeded: {global_sla_ms} ms; soft-cancel")
                bb.set("route", "review")
                break
        return bb
