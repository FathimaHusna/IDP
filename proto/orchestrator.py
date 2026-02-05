import json
from pathlib import Path
from .ocr import ocr_text
from .classify import classify
from .extract import extract
from .validate import validate
from .utils import sha256_bytes
from .governance import enforce_privacy_policy, check_financial_integrity
from .router import detect_template


def score_final(class_prob: float, completeness: float, validation_score: float) -> float:
    w1, w2, w3 = 0.3, 0.4, 0.3
    return round(w1 * class_prob + w2 * completeness + w3 * validation_score, 4)


def process_file(path: str) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    doc_id = sha256_bytes(raw)
    ocr = ocr_text(path)
    text = ocr["text"]
    ocr_quality = ocr["quality"]
    clf = classify(text)
    doc_type = clf["type"]
    # Template-based overrides (e.g., job completion → receipt)
    tmpl = detect_template(text)
    template_name = tmpl.get("template", "unknown")
    force_type = tmpl.get("force_type")
    if force_type:
        doc_type = force_type
    # Heuristic override: job completion/delivery notes should be receipts
    lower = text.lower()
    if doc_type != "receipt" and (
        ("job completion" in lower) or ("delivery note" in lower) or ("completion certificate" in lower) or ("qccd" in lower)
    ):
        doc_type = "receipt"
    probs = clf["probs"]
    payload = extract(doc_type, text)
    val = validate(doc_type, payload)
    # Boost class probability when router confidently forces a type
    type_prob = probs.get(doc_type, 0.0)
    if force_type:
        type_prob = max(type_prob, 0.8)
    final_conf = score_final(type_prob, val["completeness"], val["validation_score"])
    # Anchor bonuses: reward strong cross-field evidence
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
    final_conf = min(1.0, final_conf + bonus)
    # Per-type routing thresholds: receipts tend to have fewer fields
    if doc_type == "receipt":
        route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.6 else "reject")
    else:
        route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.7 else "reject")
    redacted_text, pii_stats = enforce_privacy_policy(text)
    fraud = None
    if doc_type == "invoice":
        ok, msg = check_financial_integrity(payload.get("subtotal"), payload.get("tax"), payload.get("total"))
        fraud = {"ok": ok, "message": msg}
    return {
        "docId": doc_id,
        "file": str(p),
        "type": doc_type,
        "class_probs": probs,
        "ocr_quality": ocr_quality,
        "extraction": payload,
        "validation": val,
        "governance": {
            "privacy": {
                "emails_redacted": pii_stats.get("emails_redacted", 0),
                "phones_redacted": pii_stats.get("phones_redacted", 0),
                "redacted_preview": redacted_text[:300],
            },
            "fraud_check": fraud,
        },
        "final_confidence": final_conf,
        "route": route,
        "template": template_name,
    }


def process_folder(folder: str) -> list[dict]:
    results = []
    for f in sorted(Path(folder).glob("*")):
        if f.is_file():
            try:
                results.append(process_file(str(f)))
            except Exception as e:
                results.append({"file": str(f), "error": str(e)})
    return results


def to_json(o: dict) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)
