from .schemas import INVOICE_SCHEMA, PO_SCHEMA, CONTRACT_SCHEMA, RECEIPT_SCHEMA


def _completeness(required: list[str], payload: dict) -> float:
    if not required:
        return 1.0
    filled = 0
    for k in required:
        v = payload.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        filled += 1
    return filled / len(required)


def validate_invoice(payload: dict) -> dict:
    req = INVOICE_SCHEMA["required"]
    completeness = _completeness(req, payload)
    subtotal = payload.get("subtotal")
    tax = payload.get("tax")
    total = payload.get("total")
    math_ok = False
    if subtotal is not None and tax is not None and total is not None:
        math_ok = abs((subtotal + tax) - total) <= 0.01
    val_score = 1.0 if math_ok else 0.0
    return {
        "completeness": completeness,
        "validation_score": val_score,
        "rules": {"totals_check": math_ok},
    }


def validate_po(payload: dict) -> dict:
    req = PO_SCHEMA["required"]
    completeness = _completeness(req, payload)
    return {"completeness": completeness, "validation_score": 1.0, "rules": {}}


def validate_contract(payload: dict) -> dict:
    req = CONTRACT_SCHEMA["required"]
    completeness = _completeness(req, payload)
    return {"completeness": completeness, "validation_score": 1.0, "rules": {}}


def validate(doc_type: str, payload: dict) -> dict:
    if doc_type == "invoice":
        return validate_invoice(payload)
    if doc_type == "po":
        return validate_po(payload)
    if doc_type == "contract":
        return validate_contract(payload)
    if doc_type == "receipt":
        req = RECEIPT_SCHEMA["required"]
        return {"completeness": _completeness(req, payload), "validation_score": 1.0, "rules": {}}
    return {"completeness": 0.0, "validation_score": 0.0, "rules": {}}
