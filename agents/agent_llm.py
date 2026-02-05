from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .agent_core import Agent, Blackboard
from common.llm import LLMClient, parse_json_block
from proto.validate import validate


EXTRACT_SYSTEM = (
    "You are a precise information extraction assistant. "
    "Return only a compact JSON object matching the requested schema; no commentary."
)


def _schema_for(doc_type: str) -> str:
    if doc_type == "invoice":
        return (
            '{"vendor": str|null, "invoice_number": str|null, "subtotal": float|null, '
            '"tax": float|null, "total": float|null, "lpo_ref": str|null, "currency": str|null, '
            '"payment_received": float|null}'
        )
    if doc_type == "po":
        return (
            '{"vendor": str|null, "po_number": str|null, "lpo_ref": str|null, '
            '"total_amount": float|null, "currency": str|null}'
        )
    if doc_type == "receipt":
        return (
            '{"vendor": str|null, "reference": str|null, "lpo_ref": str|null, '
            '"invoice_ref": str|null, "amount": float|null, "currency": str|null}'
        )
    return "{}"


class LLMExtractAgent(Agent):
    name = "llm_extract"

    def __init__(self) -> None:
        self.client = LLMClient()

    def run(self, bb: Blackboard) -> None:
        # Only run when enabled and we need help (low completeness or math issue)
        if not self.client.enabled:
            return
        doc_type = bb.get("doc_type")
        text = bb.get("ocr_text", "")
        val = bb.get("validation", {})
        completeness = float(val.get("completeness", 1.0))
        math_ok = bool((val.get("rules") or {}).get("totals_check", True))
        if completeness >= 0.75 and math_ok:
            return
        schema = _schema_for(doc_type)
        user = (
            f"Document type: {doc_type}.\n"
            f"Extract fields and return JSON matching this schema: {schema}.\n"
            f"Text:\n" + text[:8000]
        )
        try:
            out = self.client.chat([
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ])
            json_obj = parse_json_block(out) or {}
            if json_obj:
                pay = dict(bb.get("extraction", {}))
                pay.update({k: v for k, v in json_obj.items() if k is not None})
                bb.set("extraction", pay)
                bb.set("validation", validate(doc_type, pay))
                logs = bb.logs
                logs.append("✓ llm_extract: applied improved payload")
        except Exception as e:
            bb.logs.append(f"✗ llm_extract: {e}")


class LLMPlannerAgent(Agent):
    name = "llm_planner"

    def __init__(self) -> None:
        self.client = LLMClient()

    def run(self, bb: Blackboard) -> None:
        if not self.client.enabled:
            return
        # Ask LLM to choose next safe actions given current observation
        doc_type = bb.get("doc_type")
        val = bb.get("validation", {})
        reasons = bb.get("reason_codes", []) or []
        ocr_q = bb.get("ocr_quality")
        obs = {
            "doc_type": doc_type,
            "validation": val,
            "reason_codes": reasons,
            "ocr_quality": ocr_q,
        }
        prompt = (
            "You decide safe next actions for an IDP pipeline. Options: "
            "re_ocr_fast, alt_extract, skip_crossdoc, proceed. "
            "Return a JSON like {\"actions\":[\"re_ocr_fast\"]}." 
            f"Observation: {obs}"
        )
        try:
            out = self.client.chat([
                {"role": "system", "content": "You are a concise planner. Only return JSON with actions."},
                {"role": "user", "content": prompt},
            ])
            obj = parse_json_block(out) or {}
            actions: List[str] = obj.get("actions") or []
            trace = bb.get("react_trace", [])
            for a in actions:
                if a == "re_ocr_fast":
                    bb.set("_planner_re_ocr_fast", True)
                    trace.append({"thought": "LLM suggests fast re-OCR", "action": a, "observation": "queued"})
                if a == "alt_extract":
                    bb.set("_planner_alt_extract", True)
                    trace.append({"thought": "LLM suggests alt extract", "action": a, "observation": "queued"})
                if a == "skip_crossdoc":
                    bb.set("_skip_crossdoc", True)
                    trace.append({"thought": "LLM suggests skipping cross-doc", "action": a, "observation": "set"})
                if a == "proceed":
                    trace.append({"thought": "LLM suggests proceed", "action": a, "observation": "noop"})
            bb.set("react_trace", trace)
        except Exception as e:
            bb.logs.append(f"✗ llm_planner: {e}")

