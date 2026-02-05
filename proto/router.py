import re


def detect_template(text: str) -> dict:
    t = text.lower()
    # MTC: Job completion / delivery note with QCDD cues
    if re.search(r"job\s*completion|delivery\s*note|completion\s*certificate|qccd", t):
        return {"template": "mtc_receipt", "force_type": "receipt"}

    # Al Mirza: Proforma/Invoice with pipe-prefixed totals and AL MIRZA TRADING header
    if ("al mirza trading" in t or "almirza" in t) and re.search(r"\|\|\s*subtotal|proforma\s+value", t):
        return {"template": "al_mirza_invoice", "force_type": "invoice"}

    # ORYXI / Manycon invoice: invoice number and total due; Manycon vendor present
    if ("manycon trading" in t or "oryxi maintenance" in t) and re.search(r"invoice\s*(number|no|#)|total\s*due|total\s+qar|total\s+value", t):
        return {"template": "oryxi_manycon_invoice", "force_type": "invoice"}

    # Qasr Al Abwab invoice: final payment line and QASR header
    if ("qasr al abwab" in t) and re.search(r"final\s*payment|amount\s*chargeable|advance\s*received|total\s*amount", t):
        return {"template": "qasr_invoice", "force_type": "invoice"}

    # Generic INV- pattern
    if re.search(r"\binv[- ]?\d{2,}", t):
        return {"template": "generic_inv", "force_type": "invoice"}

    return {"template": "unknown"}

