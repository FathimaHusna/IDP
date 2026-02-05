#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent_core import Agent, Blackboard, Orchestrator

# Reuse existing pipeline building blocks
from proto.ocr import ocr_text
from proto.classify import classify
from proto.router import detect_template
from proto.extract import extract
from proto.validate import validate
from proto.governance import enforce_privacy_policy, check_financial_integrity
from proto.orchestrator import score_final
from proto.utils import sha256_bytes
from proto.payload_models import validate_payload_schema
from common.policy import load_policy
from common.security import antivirus_scan_path, pdf_page_count_from_path
from proto.router import detect_template as _detect_template

from .reconcile import normalize_input as recon_normalize_input, group_docs as recon_group_docs
import os
from .agent_llm import LLMExtractAgent, LLMPlannerAgent


class ReconSession:
    """Holds cross-document context within a run."""
    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        # lightweight memory for vendors and LPO refs seen so far
        self.vendor_set: set[str] = set()
        self.lpo_seen: set[str] = set()
        self.text_cache: List[str] = []  # optional, keep small text snippets for retrieval

    def add_doc(self, doc: Dict[str, Any]) -> None:
        self.docs.append(doc)
        v = doc.get("vendor")
        if v:
            self.vendor_set.add(str(v).strip())
        lpo = doc.get("lpo_ref") or doc.get("po_number")
        if lpo:
            self.lpo_seen.add(str(lpo).strip())


class ExemplarRetriever:
    """Lightweight local retriever over session docs.

    Prefers same LPO or vendor; optional spaCy vector similarity if a model is present.
    """

    def __init__(self) -> None:
        self._nlp = None  # lazy-loaded only if explicitly enabled

    def _maybe_load_nlp(self) -> None:
        if self._nlp is not None:
            return
        if os.environ.get("IDP_ENABLE_SPACY", "0") != "1":
            return
        try:
            import spacy  # type: ignore
            for name in ("en_core_web_sm", "en_core_web_md", "xx_sent_ud_sm"):
                try:
                    self._nlp = spacy.load(name)  # type: ignore
                    break
                except Exception:
                    continue
        except Exception:
            self._nlp = None

    def _sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        # Prefer vectors if available
        self._maybe_load_nlp()
        if self._nlp is not None:
            try:
                va = self._nlp(a).vector
                vb = self._nlp(b).vector
                import numpy as _np  # type: ignore

                na = _np.linalg.norm(va) + 1e-9
                nb = _np.linalg.norm(vb) + 1e-9
                return float(_np.dot(va, vb) / (na * nb))
            except Exception:
                pass
        # Fallback: token Jaccard similarity
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))

    def get_exemplars(
        self,
        session: Optional[ReconSession],
        cur_text: str,
        cur_vendor: Optional[str],
        cur_lpo: Optional[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not isinstance(session, ReconSession) or not session.docs:
            return []
        # First filter by LPO then vendor; if none, rank by similarity of raw text if available (best-effort)
        pool = list(session.docs)
        if cur_lpo:
            pool = [d for d in pool if (d.get("lpo_ref") == cur_lpo or d.get("po_number") == cur_lpo)] or pool
        if cur_vendor:
            pool2 = [d for d in pool if d.get("vendor") == cur_vendor]
            pool = pool2 or pool
        # Rank by similarity using text if available; otherwise return first few
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for d in pool:
            txt = ""
            # We don't keep the full OCR text in session; best-effort is to combine keys as a proxy
            txt = " ".join(str(d.get(k) or "") for k in ("vendor", "po_number", "lpo_ref", "invoice_number", "reference"))
            s = self._sim(cur_text, txt)
            scored.append((s, d))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:top_k]]

class IngestAgent(Agent):
    name = "ingest"

    def run(self, bb: Blackboard) -> None:
        path = bb.get("path")
        p = Path(path)
        raw = p.read_bytes()
        bb.set("doc_id", sha256_bytes(raw))
        # Antivirus hook (best-effort)
        clean, details = antivirus_scan_path(str(p))
        if not clean:
            bb.log("AV flagged file; routing to reject")
            bb.set("reason_codes", sorted(set((bb.get("reason_codes") or []) + ["virus_detected"])))
            bb.set("route", "reject")
            # Continue, but skip OCR
            bb.set("ocr_text", "")
            bb.set("ocr_quality", 0.0)
            return
        # Page limits
        try:
            if p.suffix.lower() == ".pdf":
                pages = pdf_page_count_from_path(str(p))
                if pages is not None:
                    bb.set("page_count", pages)
        except Exception:
            pass
        ocr = ocr_text(str(p))
        bb.set("ocr_text", ocr.get("text", ""))
        bb.set("ocr_quality", ocr.get("quality", 0.6))


class ClassifyAgent(Agent):
    name = "classify"

    def run(self, bb: Blackboard) -> None:
        text = bb.get("ocr_text", "")
        clf = classify(text)
        doc_type = clf["type"]
        router = detect_template(text)
        if router.get("force_type"):
            doc_type = router["force_type"]
        bb.set("doc_type", doc_type)
        bb.set("class_probs", clf["probs"]) 
        bb.set("template", router.get("template", "unknown"))


class ExtractAgent(Agent):
    name = "extract"

    def run(self, bb: Blackboard) -> None:
        text = bb.get("ocr_text", "")
        doc_type = bb.get("doc_type")
        payload = extract(doc_type, text)
        # Graph-aware nudges: fill LPO and Vendor when safe matches found in text
        try:
            sess: Optional[ReconSession] = bb.get("_session")
            if doc_type in {"invoice", "receipt"} and payload.get("lpo_ref") in (None, "") and isinstance(sess, ReconSession):
                # Try to find any seen LPO tokens appearing in text
                cand = None
                for seen in list(sess.lpo_seen)[:50]:  # cap to avoid long loops
                    if seen and seen in text:
                        cand = seen
                        break
                if cand:
                    payload["lpo_ref"] = cand
            if not payload.get("vendor") and isinstance(sess, ReconSession):
                vmatch = None
                for v in list(sess.vendor_set)[:50]:
                    if v and v.lower() in text.lower():
                        vmatch = v
                        break
                if vmatch:
                    payload["vendor"] = vmatch
        except Exception:
            pass
        bb.set("extraction", payload)
        # Mark initial attempt
        bb.set("_extract_attempt", "primary")


class ExemplarsAgent(Agent):
    name = "exemplars"

    def __init__(self) -> None:
        self.retriever = ExemplarRetriever()

    def run(self, bb: Blackboard) -> None:
        sess: Optional[ReconSession] = bb.get("_session")
        text = bb.get("ocr_text", "")
        payload = bb.get("extraction", {})
        exs = self.retriever.get_exemplars(sess, text, payload.get("vendor"), payload.get("lpo_ref"))
        bb.set("exemplars", exs)


class ExemplarAugmentAgent(Agent):
    name = "exemplar_augment"

    def run(self, bb: Blackboard) -> None:
        payload = dict(bb.get("extraction", {}))
        exs: List[Dict[str, Any]] = bb.get("exemplars", []) or []
        changed = False
        # If currency missing, use dominant currency from exemplars if unambiguous
        if not payload.get("currency") and exs:
            currs = [e.get("currency") for e in exs if e.get("currency")]
            if currs:
                from collections import Counter

                c = Counter(currs)
                top, cnt = c.most_common(1)[0]
                if top and cnt >= 2:  # require at least 2 exemplars agreeing
                    payload["currency"] = top
                    changed = True
        if changed:
            bb.set("extraction", payload)


class ValidateAgent(Agent):
    name = "validate"

    def run(self, bb: Blackboard) -> None:
        doc_type = bb.get("doc_type")
        payload = bb.get("extraction", {})
        val = validate(doc_type, payload)
        bb.set("validation", val)

        # Retry strategy: invoice totals/fields recovery when math fails or completeness is low
        try:
            if doc_type == "invoice":
                rules = val.get("rules", {})
                completeness = float(val.get("completeness", 0.0))
                math_ok = bool(rules.get("totals_check", True))
                needs_retry = (not math_ok) or (completeness < 0.6)
                pay = dict(payload)
                if needs_retry:
                    # Attempt fallback using line items and alternative recomputation
                    items = pay.get("line_items") or []
                    try:
                        from math import fsum
                        subtotal_items = fsum([float(i.get("line_total") or 0.0) for i in items]) if items else None
                    except Exception:
                        subtotal_items = None

                    # If subtotal missing or mismatched, prefer items sum when reasonable
                    if subtotal_items is not None and (pay.get("subtotal") is None or (pay.get("total") and pay.get("tax") is not None and abs((subtotal_items + (pay.get("tax") or 0.0)) - pay.get("total")) <= 0.01)):
                        pay["subtotal"] = round(subtotal_items, 2)

                    # If tax missing but subtotal and total present, infer tax
                    if pay.get("tax") is None and (pay.get("subtotal") is not None and pay.get("total") is not None):
                        inferred_tax = float(pay.get("total") or 0.0) - float(pay.get("subtotal") or 0.0)
                        if abs(inferred_tax) <= max(0.01, 0.005 * max(1.0, float(pay.get("total") or 1.0))):
                            pay["tax"] = round(inferred_tax, 2)

                    # If total missing but subtotal and tax present, compute total
                    if pay.get("total") is None and (pay.get("subtotal") is not None and pay.get("tax") is not None):
                        pay["total"] = round(float(pay.get("subtotal") or 0.0) + float(pay.get("tax") or 0.0), 2)

                    # If invoice_number still missing, try a looser regex scan here as a last resort
                    if not pay.get("invoice_number"):
                        import re
                        text = bb.get("ocr_text", "")
                        mm = re.search(r"\bINV[\- ]?([A-Za-z0-9\-]{3,})\b", text, re.IGNORECASE)
                        if mm:
                            pay["invoice_number"] = mm.group(0).strip()

                    # If we made any changes, re-validate and store
                    if pay != payload:
                        bb.set("extraction", pay)
                        val2 = validate(doc_type, pay)
                        bb.set("validation", val2)
                        bb.set("_extract_attempt", "retry_totals")
        except Exception as _:
            # Retry is best-effort; keep original results if anything goes wrong
            pass


class GovernanceAgent(Agent):
    name = "governance"

    def run(self, bb: Blackboard) -> None:
        text = bb.get("ocr_text", "")
        red, pii_stats = enforce_privacy_policy(text)
        bb.set("privacy", {"redacted_preview": red[:300], **pii_stats})
        doc_type = bb.get("doc_type")
        payload = bb.get("extraction", {})
        fraud = None
        if doc_type == "invoice":
            ok, msg = check_financial_integrity(payload.get("subtotal"), payload.get("tax"), payload.get("total"))
            fraud = {"ok": ok, "message": msg}
        bb.set("fraud_check", fraud)


class ScoreRouteAgent(Agent):
    name = "score_route"

    def run(self, bb: Blackboard) -> None:
        doc_type = bb.get("doc_type")
        probs = bb.get("class_probs", {})
        payload = bb.get("extraction", {})
        val = bb.get("validation", {"completeness": 0.0, "validation_score": 0.0})

        # Router boost if template forced type
        tmpl = bb.get("template")
        type_prob = probs.get(doc_type, 0.0)
        if tmpl and tmpl != "unknown":
            type_prob = max(type_prob, 0.8)

        conf = score_final(type_prob, val.get("completeness", 0.0), val.get("validation_score", 0.0))
        bonus = 0.0
        if doc_type == "invoice":
            has_total = payload.get("total") is not None
            has_id = payload.get("invoice_number") is not None
            lpo = payload.get("lpo_ref")
            has_lpo = lpo is not None and str(lpo).strip().lower() not in {"", "nil"}
            if has_total and (has_id or has_lpo):
                bonus += 0.1
            if payload.get("payment_received") is not None:
                bonus += 0.05
        elif doc_type == "receipt":
            if payload.get("reference") and payload.get("lpo_ref"):
                bonus += 0.15
        elif doc_type == "po":
            if payload.get("po_number") and payload.get("vendor"):
                bonus += 0.1
        final_conf = min(1.0, conf + bonus)

        if doc_type == "receipt":
            route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.6 else "reject")
        else:
            route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.7 else "reject")

        # Nudge to review when validation looks weak
        if val.get("completeness", 0.0) < 0.5 and route == "auto_accept":
            route = "review"

        bb.set("final_confidence", final_conf)
        bb.set("route", route)


class ReconcileAgent(Agent):
    name = "reconcile"

    def run(self, bb: Blackboard) -> None:
        if bb.get("_skip_crossdoc"):
            return
        # Integrate current doc into session, compute chain status, and set reconciliation summary
        sess: Optional[ReconSession] = bb.get("_session")
        cur_file = bb.get("path")
        doc_type = bb.get("doc_type")
        payload = bb.get("extraction", {})
        if not isinstance(sess, ReconSession):
            return
        try:
            doc = recon_normalize_input({"type": doc_type, "extraction": payload}, cur_file)
            sess.add_doc(doc)
            chains = recon_group_docs(sess.docs)
            # pick the chain that includes this file in sources if possible; fallback by matching lpo_ref or invoice_number
            chosen: Optional[Dict[str, Any]] = None
            for c in chains:
                srcs = c.get("sources") or []
                if cur_file in srcs:
                    chosen = c
                    break
            if not chosen:
                lpo = doc.get("lpo_ref")
                inv = doc.get("invoice_number")
                for c in chains:
                    if lpo and (c.get("key") == lpo):
                        chosen = c
                        break
                    invs = set(c.get("invoice_numbers") or [])
                    if inv and inv in invs:
                        chosen = c
                        break
            if chosen:
                bb.set("reconciliation", {
                    "status": chosen.get("status"),
                    "anomalies": chosen.get("anomalies", []),
                    "confidence": chosen.get("confidence"),
                    "key": chosen.get("key"),
                })
        except Exception:
            # best-effort reconciliation
            pass


class ChainScoreAgent(Agent):
    name = "chain_score"

    def run(self, bb: Blackboard) -> None:
        if bb.get("_skip_crossdoc"):
            return
        rec = bb.get("reconciliation") or {}
        if not rec:
            return
        status = rec.get("status")
        anomalies = rec.get("anomalies", [])
        conf = float(bb.get("final_confidence") or 0.0)
        route = bb.get("route") or "review"
        # Adjust confidence/route based on reconciliation context
        if status in {"matched_3_way", "matched_po_invoice", "matched_invoice_receipt"} and not anomalies:
            conf = min(1.0, conf + 0.05)
        if anomalies:
            conf = max(0.0, conf - 0.1)
            if route == "auto_accept":
                route = "review"
        bb.set("final_confidence", conf)
        bb.set("route", route)


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self) -> None:
        self.policy = load_policy()

    def run(self, bb: Blackboard) -> None:
        reasons = bb.get("reason_codes", []) or []
        doc_type = bb.get("doc_type")
        payload = bb.get("extraction", {})
        ocr_q = float(bb.get("ocr_quality") or 0.0)
        route = bb.get("route") or "review"
        final_conf = float(bb.get("final_confidence") or 0.0)

        # Schema validation (guardrails)
        errs = validate_payload_schema(doc_type, payload)
        for e in errs:
            reasons.append(f"schema:{e}")

        # OCR quality gates
        min_qa = float(self.policy.get("ocr", {}).get("min_quality_auto_accept", 0.75))
        min_qp = float(self.policy.get("ocr", {}).get("min_quality_process", 0.3))
        if ocr_q < min_qp:
            reasons.append("ocr_too_low")
            route = "reject"
        elif ocr_q < min_qa and route == "auto_accept":
            route = "review"

        # Vendor allow/deny
        vendor = payload.get("vendor")
        deny = set(self.policy.get("vendors", {}).get("deny", []) or [])
        allow = set(self.policy.get("vendors", {}).get("allow", []) or [])
        if vendor and deny and str(vendor) in deny:
            reasons.append("vendor_denied")
            route = "reject"
        # If allowlist present, non-listed vendors go to review
        if vendor and allow and str(vendor) not in allow and route == "auto_accept":
            route = "review"

        # Anomalies enforcement
        rec = bb.get("reconciliation") or {}
        anoms = set((rec.get("anomalies") or []) + bb.get("reason_codes", []))
        force_rev = set(self.policy.get("anomalies", {}).get("force_review", []) or [])
        if anoms & force_rev and route == "auto_accept":
            route = "review"

        # Confidence thresholds (final clamp)
        thr = self.policy.get("routing", {}).get("auto_accept_min_confidence", {})
        default_thr = float(thr.get("default", 0.9))
        dt_thr = float(thr.get(doc_type, default_thr))
        if route == "auto_accept" and final_conf < dt_thr:
            route = "review"

        bb.set("route", route)
        if reasons:
            bb.set("reason_codes", sorted(set(reasons)))


class PlannerAgent(Agent):
    name = "planner"

    def run(self, bb: Blackboard) -> None:
        trace = []

        def step(thought: str, action: str, observation: str) -> None:
            trace.append({"thought": thought, "action": action, "observation": observation})

        ocr_q = float(bb.get("ocr_quality") or 0.0)
        doc_type = bb.get("doc_type")
        text = bb.get("ocr_text", "")
        val = bb.get("validation", {})
        rules = val.get("rules", {})

        # Re-run router if template unknown
        if not bb.get("template") or bb.get("template") == "unknown":
            step("Template unknown; retry router", "detect_template", "retry")
            try:
                t = _detect_template(bb.get("ocr_text", ""))
                if t.get("template") and t.get("template") != "unknown":
                    bb.set("template", t["template"])
                    step("Router found template", "set(template)", t.get("template"))
            except Exception as e:
                step("Router retry failed", "noop", str(e))

        # Note invoice totals status (retry may have happened in ValidateAgent)
        if doc_type == "invoice":
            if rules.get("totals_check") is False:
                step("Invoice totals mismatch persists", "retry_totals", "still mismatched")
            else:
                step("Invoice totals consistent", "noop", "ok")

        # Nudge to review when OCR is low
        if ocr_q < 0.5 and bb.get("route") == "auto_accept":
            bb.set("route", "review")
            step("Low OCR; conservative route", "set(route=review)", f"ocr_q={ocr_q:.2f}")

        # Fast re-OCR first page when text is empty/very short and SLA is tight
        try:
            consumed_ms = sum(int(e.get("duration_ms") or 0) for e in (bb.events or []))
            sla_ms = int(os.environ.get("IDP_GLOBAL_SLA_MS", "12000"))
            margin_ms = 1500
            if (not text or len(text) < 50) and (consumed_ms < sla_ms - margin_ms):
                step("Text is empty/short; try fast first-page OCR", "ocr_text(IDP_OCR_FAST=1)", "retry")
                import os as _os
                from proto.ocr import ocr_text as _ocr_text
                path = bb.get("path")
                old = _os.environ.get("IDP_OCR_FAST")
                _os.environ["IDP_OCR_FAST"] = "1"
                try:
                    r = _ocr_text(path)
                    new_text = r.get("text") or ""
                    if len(new_text) > len(text):
                        bb.set("ocr_text", new_text)
                        bb.set("ocr_quality", r.get("quality", ocr_q))
                        step("Fast OCR succeeded", "set(ocr_text)", f"len={len(new_text)}")
                finally:
                    if old is None:
                        _os.environ.pop("IDP_OCR_FAST", None)
                    else:
                        _os.environ["IDP_OCR_FAST"] = old
        except Exception as e:
            step("Fast OCR attempt failed", "noop", str(e))

        # Alternative extraction path for certain templates (e.g., proforma)
        try:
            tmpl = (bb.get("template") or "").lower()
            if doc_type == "invoice" and ("proforma" in tmpl or "pro-forma" in tmpl):
                from proto.extract import extract as _extract
                from proto.validate import validate as _validate
                norm_text = text.replace("PROFORMA VALUE", "TOTAL").replace("Proforma Value", "Total")
                pay2 = _extract(doc_type, norm_text)
                val2 = _validate(doc_type, pay2)
                if (val2.get("rules", {}).get("totals_check") is True) and (val.get("rules", {}).get("totals_check") is not True):
                    bb.set("extraction", pay2)
                    bb.set("validation", val2)
                    step("Template=proforma; alt extract improved totals", "extract(normalized)", "applied")
        except Exception as e:
            step("Alt extract failed", "noop", str(e))

        # SLA-aware: skip cross-doc if close to SLA
        try:
            consumed_ms = sum(int(e.get("duration_ms") or 0) for e in (bb.events or []))
            sla_ms = int(os.environ.get("IDP_GLOBAL_SLA_MS", "12000"))
            if consumed_ms > sla_ms - 1000:
                bb.set("_skip_crossdoc", True)
                step("Near SLA budget; skip cross-doc", "set(_skip_crossdoc=1)", f"consumed={consumed_ms}")
        except Exception:
            pass

        bb.set("react_trace", trace)

class ExceptionHandlingAgent(Agent):
    name = "exception_handling"

    def run(self, bb: Blackboard) -> None:
        reasons = []
        # OCR quality
        ocr_q = bb.get("ocr_quality")
        if ocr_q is not None and ocr_q < 0.5:
            reasons.append("low_ocr")

        doc_type = bb.get("doc_type")
        payload = bb.get("extraction", {})
        val = bb.get("validation", {})
        completeness = float(val.get("completeness", 0.0))
        if completeness < 0.5:
            reasons.append("low_completeness")

        rules = val.get("rules", {})
        if doc_type == "invoice":
            if rules.get("totals_check") is False:
                reasons.append("totals_mismatch")
            if not payload.get("total"):
                reasons.append("missing_total")
            if not payload.get("invoice_number"):
                reasons.append("missing_invoice_number")
        elif doc_type == "po":
            if not payload.get("po_number"):
                reasons.append("missing_po_number")
            if not payload.get("vendor"):
                reasons.append("missing_vendor")
            if not (payload.get("total") or payload.get("total_amount")):
                reasons.append("missing_total")
        elif doc_type == "receipt":
            if not payload.get("reference"):
                reasons.append("missing_reference")
            if payload.get("amount") in (None, ""):
                reasons.append("missing_amount")

        if reasons:
            bb.set("reason_codes", reasons)
            # Ensure conservative routing
            if bb.get("route") == "auto_accept":
                bb.set("route", "review")


def run_on_path(path: str, session: Optional[ReconSession] = None, use_llm: Optional[bool] = None) -> Dict[str, Any]:
    bb = Blackboard({"path": path, "_session": session})
    enable_crossdoc = os.environ.get("IDP_ENABLE_CROSSDOC", "1") == "1"
    if use_llm is None:
        use_llm = os.environ.get("LLM_ENABLED", "0") == "1"
    agents: List[Agent] = [
        IngestAgent(),
        ClassifyAgent(),
        ExtractAgent(),
        ValidateAgent(),
        GovernanceAgent(),
        ScoreRouteAgent(),
        ExceptionHandlingAgent(),
    ]
    if enable_crossdoc:
        # Insert cross-document agents just before final exception handling
        agents = [
            IngestAgent(),
            ClassifyAgent(),
            ExtractAgent(),
            *( [LLMExtractAgent()] if use_llm else [] ),
            ExemplarsAgent(),
            ExemplarAugmentAgent(),
            ValidateAgent(),
            GovernanceAgent(),
            ScoreRouteAgent(),
            PlannerAgent(),
            *( [LLMPlannerAgent()] if use_llm else [] ),
            ReconcileAgent(),
            ChainScoreAgent(),
            PolicyAgent(),
            ExceptionHandlingAgent(),
        ]
    else:
        # Lightweight pipeline with policy checks
        agents = [
            IngestAgent(),
            ClassifyAgent(),
            ExtractAgent(),
            *( [LLMExtractAgent()] if use_llm else [] ),
            ValidateAgent(),
            GovernanceAgent(),
            ScoreRouteAgent(),
            *( [LLMPlannerAgent()] if use_llm else [] ),
            PlannerAgent(),
            PolicyAgent(),
            ExceptionHandlingAgent(),
        ]
    orch = Orchestrator(agents)
    bb = orch.execute(bb)
    return {
        "docId": bb.get("doc_id"),
        "file": str(path),
        "type": bb.get("doc_type"),
        "class_probs": bb.get("class_probs", {}),
        "ocr_quality": bb.get("ocr_quality"),
        "extraction": bb.get("extraction", {}),
        "validation": bb.get("validation", {}),
        "governance": {
            "privacy": bb.get("privacy"),
            "fraud_check": bb.get("fraud_check"),
        },
        "final_confidence": bb.get("final_confidence"),
        "route": bb.get("route"),
        "template": bb.get("template"),
        "reconciliation": bb.get("reconciliation"),
        "react_trace": bb.get("react_trace", []),
        "reason_codes": bb.get("reason_codes", []),
        "agent_events": bb.events,
        "agent_logs": bb.logs,
    }


def run_folder(folder: str, use_llm: Optional[bool] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    session = ReconSession()
    for p in sorted(Path(folder).glob("*")):
        if p.is_file():
            try:
                out.append(run_on_path(str(p), session=session, use_llm=use_llm))
            except Exception as e:
                out.append({"file": str(p), "error": str(e)})
    return out


def run_folder_with_recon(folder: str, use_llm: Optional[bool] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    session = ReconSession()
    results: List[Dict[str, Any]] = []
    for p in sorted(Path(folder).glob("*")):
        if p.is_file():
            try:
                results.append(run_on_path(str(p), session=session, use_llm=use_llm))
            except Exception as e:
                results.append({"file": str(p), "error": str(e)})
    # Build chains from session docs
    try:
        chains = recon_group_docs(session.docs)
    except Exception:
        chains = []
    return results, chains


def main():
    ap = argparse.ArgumentParser(description="Agentic Orchestrator for local IDP")
    ap.add_argument("path", help="File or folder")
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        res = run_folder(str(p))
        for r in res:
            print(json.dumps(r, ensure_ascii=False))
    else:
        r = run_on_path(str(p))
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
