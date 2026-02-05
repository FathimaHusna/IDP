#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from datetime import datetime, timedelta


PROJECTS = [
    "Katara Metro Station",
    "Warehouse Refurbishment – Street 41 (Al Asmakh)",
    "Najma Office Fitout",
    "Doha Showroom Upgrade",
]

DESCRIPTIONS = [
    "FM 200 integrity testing (1-QCDD)",
    "Supply & installation of non-fire rated wood door with HW set",
    "Supply & installation of fire rated wood door with HW set",
    "Glass door with wooden frame (lumpsum)",
]

PAYMENT_TERMS = [
    "100% upon completion",
    "40% advance 60% upon delivery",
    "40% advance 55% progress 5% retention",
]


def load_vendors(path):
    vendors = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vendors.append(row)
    # Filter out bank/buyer meta rows
    return [v for v in vendors if "Bank" not in v["vendor_name"] and "Buyer" not in v["vendor_name"]]


def qdate(base, offset_days=0):
    d = base + timedelta(days=offset_days)
    # Mix formats like "12-OCT-2023" and "January 4, 2026"
    if random.random() < 0.5:
        return d.strftime("%d-%b-%Y").upper()
    else:
        return d.strftime("%B %d, %Y")


def money(amount):
    return {
        "currency": "QAR",
        "value": round(float(amount), 2),
        "text": f"QAR {float(amount):,.2f}",
    }


def ensure_dirs(out_root):
    for sub in ("pos", "invoices", "receipts", "chains"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def gen_ids(i):
    y = 2026
    po_no = f"OMS-{y}-PO-{598 + (i % 300):04d}"
    inv_no = f"{5800 + i}"
    ref = f"{3200 + i}"
    job_no = f"JO-{6000 + i:04d}.{random.randint(0,2):02d}"
    return po_no, inv_no, ref, job_no


def gen_chain(i, buyer_name, vendor):
    base_date = datetime(2026, 1, 4) + timedelta(days=random.randint(0, 30))
    po_no, inv_no, qtn_ref, job_no = gen_ids(i)
    project = random.choice(PROJECTS)
    desc = random.choice(DESCRIPTIONS)
    terms = random.choice(PAYMENT_TERMS)

    # Amount model: base +/- noise, optional discount/advance/retention
    unit_price = random.choice([500.0, 1000.0, 3500.0, 14700.0])
    qty = random.choice([1, 1, 2])
    subtotal = unit_price * qty
    tax_rate = random.choice([0.0, 0.05])  # Qatar VAT currently 0, but allow 5% for stress
    tax = round(subtotal * tax_rate, 2)
    total = subtotal + tax

    # Simulate advance/discounts depending on terms
    advance = 0.0
    discount = 0.0
    retention = 0.0
    if "advance 60%" in terms:
        advance = round(total * 0.40, 2)
    if "retention" in terms:
        advance = round(total * 0.40, 2)
        retention = round(total * 0.05, 2)

    # PO
    po_doc = {
        "doc_id": f"po_{i}",
        "doc_type": "purchase_order",
        "po_number": po_no,
        "buyer": buyer_name,
        "vendor": vendor["vendor_name"],
        "project": project,
        "date": qdate(base_date, 0),
        "currency": "QAR",
        "payment_terms": terms,
        "items": [
            {"description": desc, "qty": qty, "unit": random.choice(["Lot", "Nos"]), "unit_price": unit_price, "amount": subtotal}
        ],
        "totals": {"subtotal": subtotal, "tax": tax, "total": total},
        "references": {"quote_ref": qtn_ref},
    }

    # Invoice (can be proforma or tax invoice; allow bilingual cue)
    invoice_type = random.choice(["TAX INVOICE", "PROFORMA INVOICE", "فاتورة ضريبية"])
    inv_doc = {
        "doc_id": f"inv_{i}",
        "doc_type": "invoice",
        "invoice_number": inv_no,
        "invoice_type": invoice_type,
        "buyer": buyer_name,
        "vendor": vendor["vendor_name"],
        "project": project,
        "date": qdate(base_date, random.choice([4, 8])),
        "due_date": qdate(base_date, random.choice([4, 8])),
        "currency": "QAR",
        "payment_terms": terms,
        "items": [
            {"description": desc, "qty": qty, "unit": random.choice(["Lot", "Nos"]), "unit_price": unit_price, "amount": subtotal}
        ],
        "lpo_ref": po_no,
        "totals": {
            "subtotal": subtotal,
            "tax": tax,
            "discount": discount,
            "advance_received": advance,
            "retention": retention,
            "total": total,
            "amount_due": max(0.0, total - advance - discount - retention),
        },
        "bank": {
            "iban": "QA21 MAFR 0000 0000 0011 1040 9700 1",
            "swift": "MAFRQAQA",
            "bank": "ALRAYAN",
            "branch": "SALWA ROAD BRANCH",
        },
    }

    # Receipt/Job Completion Certificate
    receipt_doc = {
        "doc_id": f"rcp_{i}",
        "doc_type": "job_completion_certificate",
        "reference": f"MTC-2026-JCR-INT-{3000 + i}",
        "buyer": buyer_name,
        "vendor": vendor["vendor_name"],
        "project": project,
        "date": qdate(base_date, random.choice([6, 10, 12])),
        "work_description": "FM200 ROOM INTEGRITY TEST (1 QCDD)",
        "unit": "LUMPSUM",
        "qty": 1,
        "completion": "100% COMPLETED",
        "lpo_ref": po_no,
        "invoice_ref": inv_no,
        "amount": total,
    }

    # Graph edges
    edges = [
        {"subject": inv_doc["invoice_number"], "predicate": "references_po", "object": po_doc["po_number"], "docId": inv_doc["doc_id"]},
        {"subject": receipt_doc["reference"], "predicate": "confirms_invoice", "object": inv_doc["invoice_number"], "docId": receipt_doc["doc_id"]},
        {"subject": receipt_doc["reference"], "predicate": "references_po", "object": po_doc["po_number"], "docId": receipt_doc["doc_id"]},
        {"subject": vendor["vendor_name"], "predicate": "is_vendor_for", "object": inv_doc["invoice_number"], "docId": inv_doc["doc_id"]},
        {"subject": vendor["vendor_name"], "predicate": "is_vendor_for", "object": po_doc["po_number"], "docId": po_doc["doc_id"]},
        {"subject": vendor["vendor_name"], "predicate": "is_vendor_for", "object": receipt_doc["reference"], "docId": receipt_doc["doc_id"]},
        {"subject": project, "predicate": "has_document", "object": inv_doc["invoice_number"], "docId": inv_doc["doc_id"]},
        {"subject": project, "predicate": "has_document", "object": po_doc["po_number"], "docId": po_doc["doc_id"]},
        {"subject": project, "predicate": "has_document", "object": receipt_doc["reference"], "docId": receipt_doc["doc_id"]},
    ]

    return po_doc, inv_doc, receipt_doc, edges


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Qatar P2P chains (PO→Invoice→Job Completion)")
    ap.add_argument("--out", default="data/synthetic", help="Output root directory")
    ap.add_argument("--vendor_csv", default="data/raw/internal/vendor_master.csv", help="Vendor master CSV path")
    ap.add_argument("--buyer", default="ORYXI Maintenance Services (Buyer)", help="Buyer name")
    ap.add_argument("--n", type=int, default=50, help="Number of chains to generate")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dirs(args.out)

    vendors = load_vendors(args.vendor_csv)
    if not vendors:
        raise SystemExit("No vendors found in vendor master (excluding Buyer/Bank rows)")

    edges_all = []
    for i in range(args.n):
        vendor = random.choice(vendors)
        po, inv, rcp, edges = gen_chain(i, args.buyer, vendor)
        write_json(os.path.join(args.out, "pos", f"po_{i}.json"), po)
        write_json(os.path.join(args.out, "invoices", f"inv_{i}.json"), inv)
        write_json(os.path.join(args.out, "receipts", f"rcp_{i}.json"), rcp)
        edges_all.extend(edges)

    edges_path = os.path.join(args.out, "chains", "edges.jsonl")
    with open(edges_path, "w", encoding="utf-8") as f:
        for e in edges_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} P2P chains under {args.out}; edges: {edges_path}")


if __name__ == "__main__":
    main()
