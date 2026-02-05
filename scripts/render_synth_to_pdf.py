#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


def draw_kv(c: canvas.Canvas, x: float, y: float, k: str, v: str, bold=False):
    if bold:
        c.setFont("Helvetica-Bold", 11)
    else:
        c.setFont("Helvetica", 11)
    c.drawString(x, y, f"{k}:")
    c.setFont("Helvetica", 11)
    c.drawString(x + 90, y, str(v))


def render_invoice(doc: Dict[str, Any], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 30 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Invoice / فاتورة ضريبية")
    y -= 12 * mm

    draw_kv(c, 20 * mm, y, "Vendor", doc.get("vendor", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Invoice number", doc.get("invoice_number", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Date", doc.get("date", doc.get("due_date", ""))); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "LPO Ref", doc.get("lpo_ref", "")); y -= 7 * mm
    totals = doc.get("totals", {})
    draw_kv(c, 20 * mm, y, "Subtotal", f"QAR {totals.get('subtotal', doc.get('subtotal', ''))}"); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Tax", f"QAR {totals.get('tax', doc.get('tax', ''))}"); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Total", f"QAR {totals.get('total', doc.get('total', ''))}", bold=True); y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Items"); y -= 7 * mm
    c.setFont("Helvetica", 11)
    items = doc.get("items", [])
    for it in items[:10]:
        line = f"- {it.get('description','')} | Qty: {it.get('qty', it.get('quantity',''))} | Unit: {it.get('unit','')} | Price: {it.get('unit_price','')} | Amount: {it.get('amount', it.get('line_total',''))}"
        c.drawString(25 * mm, y, line[:100])
        y -= 6 * mm
        if y < 30 * mm:
            c.showPage(); y = h - 30 * mm

    c.showPage(); c.save()


def render_po(doc: Dict[str, Any], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 30 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Purchase Order"); y -= 12 * mm
    draw_kv(c, 20 * mm, y, "Buyer", doc.get("buyer", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Vendor", doc.get("vendor", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "PO number", doc.get("po_number", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Date", doc.get("date", doc.get("po_date", ""))); y -= 7 * mm
    totals = doc.get("totals", {})
    draw_kv(c, 20 * mm, y, "Total", f"QAR {totals.get('total', doc.get('total_amount', ''))}", bold=True); y -= 10 * mm
    c.showPage(); c.save()


def render_receipt(doc: Dict[str, Any], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 30 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Job Completion Certificate / Delivery Note"); y -= 12 * mm
    draw_kv(c, 20 * mm, y, "Reference", doc.get("reference", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Date", doc.get("date", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Client", doc.get("buyer", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "Vendor", doc.get("vendor", "")); y -= 7 * mm
    draw_kv(c, 20 * mm, y, "LPO Ref", doc.get("lpo_ref", "")); y -= 7 * mm
    if doc.get("amount") is not None:
        draw_kv(c, 20 * mm, y, "Amount", f"QAR {doc.get('amount')}", bold=True); y -= 7 * mm
    c.showPage(); c.save()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Render synthetic PO/Invoice/Receipt JSON to PDFs for OCR demos")
    ap.add_argument("--src", default="data/synthetic", help="Source root")
    ap.add_argument("--out", default="data/synthetic_pdf", help="Output PDF root")
    ap.add_argument("--limit", type=int, default=10, help="Max per type (0 = all)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "pos"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "invoices"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "receipts"), exist_ok=True)

    # POs
    po_files = sorted(glob(os.path.join(args.src, "pos", "po_*.json")))
    if args.limit > 0:
        po_files = po_files[: args.limit]
    for p in po_files:
        doc = load_json(p)
        outp = os.path.join(args.out, "pos", os.path.basename(p).replace(".json", ".pdf"))
        render_po(doc, outp)

    # Invoices
    inv_files = sorted(glob(os.path.join(args.src, "invoices", "inv_*.json")))
    if args.limit > 0:
        inv_files = inv_files[: args.limit]
    for p in inv_files:
        doc = load_json(p)
        outp = os.path.join(args.out, "invoices", os.path.basename(p).replace(".json", ".pdf"))
        render_invoice(doc, outp)

    # Receipts
    rcp_files = sorted(glob(os.path.join(args.src, "receipts", "rcp_*.json")))
    if args.limit > 0:
        rcp_files = rcp_files[: args.limit]
    for p in rcp_files:
        doc = load_json(p)
        outp = os.path.join(args.out, "receipts", os.path.basename(p).replace(".json", ".pdf"))
        render_receipt(doc, outp)

    print(f"Rendered PDFs under {args.out}")


if __name__ == "__main__":
    main()

