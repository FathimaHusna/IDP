import re
from .utils import parse_date, as_float, currency_from_symbol_or_code


def _find_date_token(s: str) -> str | None:
    """Extract a date-like token from noisy text for parsing."""
    m = re.search(
        r"(\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        s,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def extract_invoice(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full = "\n".join(lines)
    vendor = None
    m = re.search(r"(?:Vendor|Supplier)\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        vendor = m.group(1).strip()
    if vendor is None:
        m = re.search(r"make\s+all\s+cheques\s+payable\s+to\s+(.+?)(?:\.|$)", full, re.IGNORECASE)
        if m:
            vendor = m.group(1).strip()
    if vendor is None:
        # Fallback: first prominent all-caps org line with W.L.L/WLL in the header block
        header_block = lines[:12]
        for l in header_block:
            if re.search(r"\bW\.?L\.?L\.?\b|\bWLL\b", l, re.IGNORECASE) and not re.search(r"invoice|proforma|client", l, re.IGNORECASE):
                vendor = re.sub(r"^[\W\d]+", "", l.strip())
                break
    # Try vendor_master.csv for clean matches
    if vendor is None or len(vendor) < 5:
        try:
            import csv, os
            vm = os.path.join("data", "raw", "internal", "vendor_master.csv")
            if os.path.exists(vm):
                with open(vm, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        name = row.get("vendor_name", "").strip()
                        if name and name.lower() in full.lower():
                            vendor = name
                            break
        except Exception:
            pass
    inv_no = None
    for pat in [
        r"Invoice\s*(?:Number|No\.?|#)\s*[:\-]?\s*([A-Za-z0-9\- ]+)",
        r"INVOICE\s*#\s*([A-Za-z0-9\- ]+)",
        r"Invoice\s*number\s*[:\-]\s*([A-Za-z0-9\- ]+)",
    ]:
        mm = re.search(pat, full, re.IGNORECASE)
        if mm:
            inv_no = mm.group(1).strip()
            break
    # Heuristic: if we matched a short/space-separated numeric like '2 5 5', search nearby for a 4+ digit number
    if inv_no and inv_no.replace(" ", "").isdigit() and len(inv_no.replace(" ", "")) <= 3:
        for i, l in enumerate(lines):
            if re.search(r"invoice\s*(number|no\.?|#)", l, re.IGNORECASE):
                for j in range(i, min(i + 6, len(lines))):
                    m2 = re.search(r"\b\d{4,8}\b", lines[j])
                    if m2:
                        inv_no = m2.group(0)
                        break
                break
    inv_date = None
    mm = re.search(r"(?:Invoice\s*Date|DATE|Date)\s*[:\-]\s*([0-9A-Za-z,\-\./ ]+)", full, re.IGNORECASE)
    if mm:
        token = _find_date_token(mm.group(1)) or mm.group(1)
        inv_date = parse_date(token)
    due_date = None
    mm = re.search(r"Due\s*date\s*[:\-]\s*([0-9A-Za-z,\-\./ ]+)", full, re.IGNORECASE)
    if mm:
        token = _find_date_token(mm.group(1)) or mm.group(1)
        due_date = parse_date(token)
    if not due_date:
        for i, l in enumerate(lines):
            if re.search(r"^\s*due\s*date\b", l, re.IGNORECASE):
                if i + 1 < len(lines):
                    token = _find_date_token(lines[i + 1]) or lines[i + 1]
                    due_date = parse_date(token)
                break
    currency = None
    mm = re.search(r"Currency\s*[:\-]\s*([A-Z]{3})", full, re.IGNORECASE)
    if mm:
        currency = currency_from_symbol_or_code(mm.group(1))
    if not currency:
        currency = currency_from_symbol_or_code(full)

    subtotal = None
    tax = None
    total = None
    # Line-wise scan: tolerate leading noise like 'Po', pipes, etc.
    for l in lines:
        if subtotal is None and re.search(r"\bSUBTOTAL\b", l, re.IGNORECASE):
            subtotal = as_float(l)
        if total is None and re.search(r"\bTOTAL\s*(?:DUE)?\b", l, re.IGNORECASE):
            total = as_float(l)
        if total is None and re.search(r"\bTOTAL\s+AMOUNT\b", l, re.IGNORECASE):
            total = as_float(l)
        if total is None and re.search(r"\bPROFORMA\s+VALUE\b", l, re.IGNORECASE):
            total = as_float(l)
        if tax is None and re.search(r"\bTAX\b", l, re.IGNORECASE):
            tax = as_float(l)

    # Optional: PAYMENT RECEIVED
    payment_received = None
    for l in lines:
        if re.search(r"PAYMENT\s+RECEIVED", l, re.IGNORECASE):
            payment_received = as_float(l)
            break
    if payment_received is None:
        for l in lines:
            if re.search(r"FINAL\s+PAYMENT", l, re.IGNORECASE):
                payment_received = as_float(l)
                break

    # LPO/PO reference: prefer explicit "LPO # X" lines; avoid capturing the label "REFERENCE"
    lpo = None
    for pat in [
        r"\bLPO\s*#\s*([A-Za-z0-9\-]+)",
        r"\bLPO\s*REF\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        r"\bCLIENT\s+LPO\s+(?:REF(?:ERENCE)?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-]+)",
        r"\b(?:Buyers?'?s?\s*)?PO\s*(?:Number|No\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
    ]:
        mm = re.search(pat, full, re.IGNORECASE)
        if mm and mm.group(1).strip().upper() not in {"REFERENCE"}:
            lpo = mm.group(1).strip()
            break
    if lpo is None:
        # If we find the label, look at the next non-empty line for the actual value
        label_idx = next((i for i, l in enumerate(lines) if re.search(r"CLIENT\s+LPO\s+REF", l, re.IGNORECASE)), None)
        if label_idx is not None:
            for j in range(label_idx + 1, min(label_idx + 4, len(lines))):
                m2 = re.search(r"\bLPO\s*#\s*([A-Za-z0-9\-]+)", lines[j], re.IGNORECASE)
                if m2:
                    lpo = m2.group(1).strip()
                    break

    # Infer tax if absent and subtotal/total present
    if tax is None and subtotal is not None and total is not None:
        diff = round(total - subtotal, 2)
        tax = diff if diff >= 0 else 0.0

    items = []
    header_idx = None
    for i, l in enumerate(lines):
        if re.search(r"Item\s*\|\s*Qty\s*\|\s*Unit Price\s*\|\s*Line Total", l, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is not None:
        for l in lines[header_idx + 1 :]:
            if re.search(r"^(Subtotal|Tax|Total)\b", l, re.IGNORECASE):
                break
            parts = [p.strip() for p in l.split("|")]
            if len(parts) == 4:
                desc, qty, unit_price, line_total = parts
                items.append(
                    {
                        "description": desc,
                        "quantity": as_float(qty),
                        "unit_price": as_float(unit_price),
                        "line_total": as_float(line_total),
                    }
                )

    return {
        "invoice_number": inv_no,
        "invoice_date": inv_date,
        "due_date": due_date,
        "vendor": vendor,
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "lpo_ref": lpo,
        "payment_received": payment_received,
        "line_items": items,
    }


def extract_po(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full = "\n".join(lines)
    buyer = None
    vendor = None
    po_no = None
    po_date = None
    currency = None
    total = None
    lpo = None
    m = re.search(r"Buyer\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        buyer = m.group(1).strip()
    m = re.search(r"Vendor\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        vendor = m.group(1).strip()
    m = re.search(r"(?:PO|LPO)\s*(?:Number|No\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)", full, re.IGNORECASE)
    if m:
        po_no = m.group(1).strip()
    m = re.search(r"(?:PO\s*Date|Date)\s*[:\-]\s*([0-9\-/]+)", full, re.IGNORECASE)
    if m:
        po_date = parse_date(m.group(1))
    m = re.search(r"Currency\s*[:\-]\s*([A-Z]{3})", full, re.IGNORECASE)
    if m:
        currency = currency_from_symbol_or_code(m.group(1))
    if not currency:
        currency = currency_from_symbol_or_code(full)
    m = re.search(r"Total\s*[:\-]?\s*([£$€]?[0-9,]+\.?[0-9]*)", full, re.IGNORECASE)
    if m:
        total = as_float(m.group(1))

    items = []
    for line in lines:
        if "|" in line and not re.search(r"Item\s*\|", line, re.IGNORECASE):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                desc, qty, unit_price, line_total = parts[:4]
                items.append(
                    {
                        "description": desc,
                        "quantity": as_float(qty),
                        "unit_price": as_float(unit_price),
                        "line_total": as_float(line_total),
                    }
                )

    return {
        "po_number": po_no,
        "po_date": po_date,
        "buyer": buyer,
        "vendor": vendor,
        "currency": currency,
        "total_amount": total,
        "lpo_ref": lpo or po_no,
        "line_items": items,
    }


def extract_receipt(text: str) -> dict:
    # Job completion certificate / delivery note
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full = "\n".join(lines)
    ref = None
    m = re.search(r"\bREF\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", full, re.IGNORECASE)
    if m:
        ref = m.group(1).strip()
    else:
        m = re.search(r"\b(?:Reference|Ref)\s*[:\-]\s*([A-Za-z0-9\-]+)", full, re.IGNORECASE)
        if m:
            ref = m.group(1).strip()
    date = None
    m = re.search(r"\bDATE\s*[:\-]\s*([0-9A-Za-z,\-\./ ]+)", full, re.IGNORECASE)
    if m:
        token = _find_date_token(m.group(1)) or m.group(1)
        date = parse_date(token)
    buyer = None
    vendor = None
    m = re.search(r"CLIENT\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        buyer = m.group(1).strip()
    m = re.search(r"(?:Company\s*Name|Vendor)\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        vendor = m.group(1).strip()
    if vendor is None:
        # Fallback: header block W.L.L/WLL line
        header_block = lines[:12]
        for l in header_block:
            if re.search(r"\bW\.?L\.?L\.?\b|\bWLL\b", l, re.IGNORECASE) and not re.search(r"invoice|proforma|client", l, re.IGNORECASE):
                vendor = re.sub(r"^[\W\d]+", "", l.strip())
                break
    lpo = None
    m = re.search(r"(?:LPO|PO)\s*REF\s*[:\-]\s*([A-Za-z0-9\-]+)", full, re.IGNORECASE)
    if m:
        lpo = m.group(1).strip()
    invoice_ref = None
    m = re.search(r"Invoice\s*(?:No\.?|#|number)\s*[:\-]?\s*([A-Za-z0-9\-]+)", full, re.IGNORECASE)
    if m:
        invoice_ref = m.group(1).strip()
    amount = None
    m = re.search(r"(?:Total\s*Amount|Total|Amount)\s*\(?(?:QAR|[A-Z]{3})?\)?\s*[:\-]?\s*([A-Za-z0-9,\. ]+)", full, re.IGNORECASE)
    if m:
        amount = as_float(m.group(1))
    return {
        "reference": ref,
        "date": date,
        "buyer": buyer,
        "vendor": vendor,
        "lpo_ref": lpo,
        "invoice_ref": invoice_ref,
        "amount": amount,
    }


def extract_contract(text: str) -> dict:
    full = text
    a = None
    b = None
    eff = None
    term = None
    law = None
    m = re.search(r"Between\s*[:\-]\s*(.+?)\s*and\s*(.+)", full, re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
    m = re.search(r"Effective\s*Date\s*[:\-]\s*([0-9\-/]+)", full, re.IGNORECASE)
    if m:
        eff = parse_date(m.group(1))
    m = re.search(r"Term\s*[:\-]\s*([0-9]+)\s*months", full, re.IGNORECASE)
    if m:
        term = int(m.group(1))
    m = re.search(r"Governing\s*Law\s*[:\-]\s*(.+)", full, re.IGNORECASE)
    if m:
        law = m.group(1).strip()
    return {
        "party_a": a,
        "party_b": b,
        "effective_date": eff,
        "term_months": term,
        "governing_law": law,
    }


def extract(doc_type: str, text: str) -> dict:
    if doc_type == "invoice":
        return extract_invoice(text)
    if doc_type == "po":
        return extract_po(text)
    if doc_type == "contract":
        return extract_contract(text)
    if doc_type == "receipt":
        return extract_receipt(text)
    return {}
