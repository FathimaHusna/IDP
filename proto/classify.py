import re


def classify(text: str) -> dict:
    t = text.lower()
    scores = {
        "invoice": 0.0,
        "po": 0.0,
        "contract": 0.0,
        "receipt": 0.0,  # job completion / delivery note
    }
    # Invoices: English, proforma, Arabic cues
    if re.search(r"\binvoice\b|proforma|فاتورة", t):
        scores["invoice"] += 1.0
    # PO / LPO cues
    if re.search(r"\bpurchase order\b|\bpo\b|\blpo\b", t):
        scores["po"] += 1.0
    # Contracts
    if re.search(r"\bcontract\b|\bagreement\b", t):
        scores["contract"] += 0.6  # reduce weight to avoid overshadowing receipt forms mentioning 'contract'
    # Receipts / job completion / delivery note cues
    if re.search(r"job\s*completion|delivery\s*note|\bgrn\b|completion\s*certificate|certificate/\s*delivery\s*note|qccd", t):
        scores["receipt"] += 1.2
        scores["contract"] -= 0.2

    # Financial anchors
    if re.search(r"subtotal|tax|total|iban|swift|qar", t):
        scores["invoice"] += 0.5
    if re.search(r"unit price|qty|quantity|amount", t):
        scores["invoice"] += 0.25
        scores["po"] += 0.25
    if re.search(r"governing law|term|parties", t):
        scores["contract"] += 0.5

    best_type = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    return {"type": best_type, "probs": probs}
