from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import LOGS_DIR


def _conn() -> sqlite3.Connection:
    dbp = LOGS_DIR / "app.db"
    dbp.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(dbp))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            agent TEXT,
            status TEXT,
            duration_ms INTEGER,
            error TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT,
            docId TEXT,
            type TEXT,
            route TEXT,
            final_confidence REAL,
            reason_codes TEXT,
            decision TEXT,
            corrections TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return con


def insert_agent_event(run_id: Optional[str], event: Dict[str, Any]) -> None:
    try:
        con = _conn()
        con.execute(
            "INSERT INTO agent_events(run_id, agent, status, duration_ms, error) VALUES(?,?,?,?,?)",
            (
                run_id,
                event.get("agent"),
                event.get("status"),
                int(event.get("duration_ms") or 0),
                event.get("error"),
            ),
        )
        con.commit()
        con.close()
    except Exception:
        pass


def insert_feedback(row: Dict[str, Any]) -> None:
    try:
        con = _conn()
        con.execute(
            "INSERT INTO feedback(file, docId, type, route, final_confidence, reason_codes, decision, corrections) VALUES(?,?,?,?,?,?,?,?)",
            (
                row.get("file"),
                row.get("docId"),
                row.get("type"),
                row.get("route"),
                float(row.get("final_confidence") or 0.0),
                ",".join(row.get("reason_codes") or []),
                row.get("decision"),
                (row.get("corrections") and str(row.get("corrections"))) or None,
            ),
        )
        con.commit()
        con.close()
    except Exception:
        pass

