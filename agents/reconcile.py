#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Prefer to import the local pipeline for non-JSON docs
try:
    from proto.orchestrator import process_file
except Exception:
    process_file = None  # type: ignore


def read_doc(path: str) -> Optional[Dict[str, Any]]:
    suf = os.path.splitext(path)[1].lower()
    if suf == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return normalize_input(obj, path)
        except Exception:
            return None
    # If this looks like a synthetic-rendered PDF (inv_X.pdf/po_X.pdf/rcp_X.pdf),
    # try to load the original JSON (better than OCR on a simple rendering).
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        prefix = base.split("_")[0]
        synth_map = {
            "po": os.path.join("data", "synthetic", "pos", base + ".json"),
            "inv": os.path.join("data", "synthetic", "invoices", base + ".json"),
            "rcp": os.path.join("data", "synthetic", "receipts", base + ".json"),
        }
        guess = synth_map.get(prefix)
        if guess and os.path.exists(guess):
            with open(guess, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return normalize_input(obj, guess)
    except Exception:
        pass

    # Fallback: process through local pipeline
    if process_file is None:
        return None
    try:
        result = process_file(path)
        ext = result.get("extraction", {})
        return {
            "source": path,
            "doc_type": result.get("type"),
            "vendor": ext.get("vendor"),
            "invoice_number": ext.get("invoice_number"),
            "po_number": ext.get("po_number"),
            "lpo_ref": ext.get("lpo_ref"),
            "subtotal": ext.get("subtotal"),
            "tax": ext.get("tax"),
            "total": ext.get("total") or ext.get("total_amount"),
            "amount": ext.get("amount"),  # for receipts
            "reference": ext.get("reference"),  # for receipts
        }
    except Exception:
        return None


def normalize_input(obj: Dict[str, Any], source: str) -> Dict[str, Any]:
    # Support synthetic docs from scripts/synth_qatar_p2p.py and orchestrator outputs
    dt = obj.get("doc_type") or obj.get("type")
    # Map synthetic types to our canonical types
    if dt == "purchase_order":
        doc_type = "po"
    elif dt in {"job_completion_certificate", "delivery_note"}:
        doc_type = "receipt"
    else:
        doc_type = dt

    # Synthetic shapes
    if doc_type == "po":
        totals = obj.get("totals", {})
        return {
            "source": source,
            "doc_type": "po",
            "vendor": obj.get("vendor"),
            "po_number": obj.get("po_number"),
            "lpo_ref": obj.get("lpo_ref") or obj.get("po_number"),
            "total": totals.get("total") or obj.get("total_amount"),
        }
    if doc_type == "invoice":
        totals = obj.get("totals", {})
        return {
            "source": source,
            "doc_type": "invoice",
            "vendor": obj.get("vendor"),
            "invoice_number": obj.get("invoice_number"),
            "lpo_ref": obj.get("lpo_ref") or obj.get("references", {}).get("po"),
            "subtotal": totals.get("subtotal"),
            "tax": totals.get("tax"),
            "total": totals.get("total"),
        }
    if doc_type == "receipt":
        return {
            "source": source,
            "doc_type": "receipt",
            "vendor": obj.get("vendor"),
            "reference": obj.get("reference"),
            "lpo_ref": obj.get("lpo_ref"),
            "invoice_ref": obj.get("invoice_ref"),
            "amount": obj.get("amount"),
        }

    # Orchestrator-like
    ext = obj.get("extraction", obj)
    return {
        "source": source,
        "doc_type": doc_type,
        "vendor": ext.get("vendor"),
        "invoice_number": ext.get("invoice_number"),
        "po_number": ext.get("po_number"),
        "lpo_ref": ext.get("lpo_ref"),
        "subtotal": ext.get("subtotal"),
        "tax": ext.get("tax"),
        "total": ext.get("total") or ext.get("total_amount"),
        "amount": ext.get("amount"),
        "reference": ext.get("reference"),
    }


def norm_vendor(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = v.lower()
    s = re.sub(r"\b(w\.?l\.?l\.?)\b", "wll", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_lpo(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = str(x).upper()
    s = s.replace("P0-", "PO-")
    s = re.sub(r"\s+REV\w*$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def norm_inv(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = re.sub(r"\s*\-\s*", "-", str(x))
    s = re.sub(r"\s+", "", s)
    return s


def group_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Index POs by LPO/PO number
    pos: Dict[str, Dict[str, Any]] = {}
    invoices: List[Dict[str, Any]] = []
    receipts: List[Dict[str, Any]] = []
    for d in docs:
        dt = d.get("doc_type")
        if dt == "po":
            key = norm_lpo(d.get("po_number") or d.get("lpo_ref"))
            if key:
                pos[key] = d
        elif dt == "invoice":
            invoices.append(d)
        elif dt == "receipt":
            receipts.append(d)

    chains: Dict[str, Dict[str, Any]] = {}
    # Match invoices to POs
    for inv in invoices:
        lpo = norm_lpo(inv.get("lpo_ref"))
        vendor = norm_vendor(inv.get("vendor"))
        po = pos.get(lpo) if lpo else None
        key = lpo or f"INV:{norm_inv(inv.get('invoice_number'))}"
        if key not in chains:
            chains[key] = {"po": po, "invoices": [], "receipts": []}
        chains[key]["invoices"].append(inv)
        chains[key]["vendor"] = vendor or chains[key].get("vendor")

    # Attach receipts
    for r in receipts:
        lpo = norm_lpo(r.get("lpo_ref"))
        inv_ref = norm_inv(r.get("invoice_ref"))
        key = lpo or (f"INV:{inv_ref}" if inv_ref else None)
        if not key:
            # orphan
            key = f"RCP:{r.get('reference')}"
        if key not in chains:
            chains[key] = {"po": pos.get(lpo) if lpo else None, "invoices": [], "receipts": []}
        chains[key]["receipts"].append(r)

    # Also include standalone POs
    for k, po in pos.items():
        chains.setdefault(k, {"po": po, "invoices": [], "receipts": []})

    # Compute statuses
    out: List[Dict[str, Any]] = []
    for key, grp in chains.items():
        po = grp.get("po")
        invs = grp.get("invoices", [])
        rcps = grp.get("receipts", [])
        anomalies: List[str] = []
        status = "partial"
        conf = 0.5

        # Vendor consistency
        vendors = [norm_vendor(x.get("vendor")) for x in ([po] if po else []) + invs + rcps]
        vendors = [v for v in vendors if v]
        vendor = max(set(vendors), key=vendors.count) if vendors else None

        # Amount checks
        po_total = po and po.get("total")
        inv_total = None
        inv_id = None
        if invs:
            # For simplicity, pick the largest total as invoice total (covers split OCR variance)
            totals = [(i.get("total"), i) for i in invs if i.get("total") is not None]
            if totals:
                inv_total, inv_best = max(totals, key=lambda t: t[0])
                inv_id = norm_inv(inv_best.get("invoice_number"))
        rcp_amount = None
        if rcps:
            amounts = [r.get("amount") for r in rcps if r.get("amount") is not None]
            if amounts:
                rcp_amount = max(amounts)

        # Linking strength
        lpo_present = key and key.startswith("OMS-")
        if lpo_present:
            conf += 0.1
        if vendor:
            conf += 0.05

        # Math check for invoice
        math_ok = True
        for inv in invs:
            st, tx, tt = inv.get("subtotal"), inv.get("tax"), inv.get("total")
            if st is not None and tx is not None and tt is not None:
                if abs((st + (tx or 0.0)) - tt) > 0.01:
                    anomalies.append("invoice_math_mismatch")
                    math_ok = False
        if invs and math_ok:
            conf += 0.05

        # PO vs Invoice
        if po_total is not None and inv_total is not None:
            if abs(po_total - inv_total) > 0.01:
                anomalies.append("po_vs_invoice_total_mismatch")
            else:
                conf += 0.1

        # Receipt confirmation
        if rcp_amount is not None and inv_total is not None:
            if abs(rcp_amount - inv_total) > 0.01:
                anomalies.append("receipt_vs_invoice_amount_mismatch")
            else:
                conf += 0.1

        # Determine status
        if po and invs and rcps and not anomalies:
            status = "matched_3_way"
        elif po and invs and not anomalies:
            status = "matched_po_invoice"
        elif invs and rcps and not anomalies:
            status = "matched_invoice_receipt"
        elif invs and not po:
            status = "invoice_only"
        elif rcps and not (po or invs):
            status = "receipt_only"

        out.append({
            "key": key,
            "vendor": vendor,
            "po_number": po and po.get("po_number"),
            "invoice_numbers": [norm_inv(i.get("invoice_number")) for i in invs if i.get("invoice_number")],
            "receipt_refs": [r.get("reference") for r in rcps if r.get("reference")],
            "po_total": po_total,
            "invoice_total": inv_total,
            "receipt_amount": rcp_amount,
            "status": status,
            "anomalies": anomalies,
            "confidence": round(max(0.0, min(1.0, conf)), 3),
            "counts": {"po": 1 if po else 0, "invoices": len(invs), "receipts": len(rcps)},
            "sources": [x.get("source") for x in ([po] if po else []) + invs + rcps],
        })

    return out


def main():
    ap = argparse.ArgumentParser(description="Reconcile PO ↔ Invoice ↔ Receipt and flag anomalies")
    ap.add_argument("path", help="Folder with JSON/PDF/TXT docs (synthetic or real)")
    ap.add_argument("--out", default="recon_report.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    files: List[str] = []
    for root, _, fns in os.walk(args.path):
        for fn in fns:
            if fn.lower().endswith((".json", ".pdf", ".png", ".jpg", ".jpeg", ".txt")):
                files.append(os.path.join(root, fn))
    docs: List[Dict[str, Any]] = []
    for f in sorted(files):
        d = read_doc(f)
        if d:
            docs.append(d)

    chains = group_docs(docs)
    with open(args.out, "w", encoding="utf-8") as fo:
        for row in chains:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Reconciled {len(chains)} chains from {len(docs)} docs → {args.out}")


if __name__ == "__main__":
    main()
