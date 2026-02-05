import json
import os
import re
from datetime import datetime


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# E.164-ish and local Gulf patterns (loose): +974 7733 3551, 3393 9634, 555-0199
PHONE_RE = re.compile(r"(?:\+\d{3,15}|\b\d{3,4})[\s-]?\d{3,4}[\s-]?\d{3,4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
SWIFT_RE = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b")


def _default_privacy_config() -> dict:
    return {
        "patterns": {
            "email": True,
            "phone": True,
            "iban": True,
            "swift": True,
            "bank_account": True,
            "qid": True,
            "pan": True,
        },
        "mask": {
            "email": "████████",
            "phone": "███-███-████",
            "iban": "████████IBAN████",
            "swift": "████SWIFT████",
            "bank_account": "████ACCOUNT████",
            "qid": "████QID████",
            "pan": "████CARD████",
        },
        "context_keywords": {
            "bank_account": ["account", "account number", "acc", "a/c", "iban", "swift"],
            "qid": ["qid", "qatar id", "id no", "id number"],
            "pan": ["card", "visa", "mastercard", "credit", "debit"],
        },
    }


def load_privacy_config(path: str = "config/privacy.yaml") -> dict:
    try:
        import yaml  # type: ignore

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
    except Exception:
        data = {}
    cfg = _default_privacy_config()
    # Shallow merge
    for k in ("patterns", "mask", "context_keywords"):
        if k in data and isinstance(data[k], dict):
            cfg[k].update(data[k])
    return cfg


def _luhn_valid(num: str) -> bool:
    s = [int(c) for c in num if c.isdigit()]
    if not (13 <= len(s) <= 19):
        return False
    checksum = 0
    parity = len(s) % 2
    for i, d in enumerate(s):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def enforce_privacy_policy(text: str, cfg: dict | None = None) -> tuple[str, dict]:
    """Redact PII based on config; returns (redacted_text, counts)."""
    cfg = cfg or load_privacy_config()
    pats = cfg.get("patterns", {})
    mask = cfg.get("mask", {})

    counts = {"emails_redacted": 0, "phones_redacted": 0, "ibans_redacted": 0, "swifts_redacted": 0, "bank_accounts_redacted": 0, "qids_redacted": 0, "pans_redacted": 0}
    red = text

    if pats.get("email", True):
        emails = EMAIL_RE.findall(red)
        counts["emails_redacted"] = len(emails)
        red = EMAIL_RE.sub(mask.get("email", "████████"), red)
    if pats.get("phone", True):
        phones = PHONE_RE.findall(red)
        counts["phones_redacted"] = len(phones)
        red = PHONE_RE.sub(mask.get("phone", "███-███-████"), red)
    if pats.get("iban", True):
        ibans = IBAN_RE.findall(red)
        counts["ibans_redacted"] = len(ibans)
        red = IBAN_RE.sub(mask.get("iban", "████████IBAN████"), red)
    if pats.get("swift", True):
        swifts = re.findall(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", red)
        counts["swifts_redacted"] = len(swifts)
        red = re.sub(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", mask.get("swift", "████SWIFT████"), red)

    # Bank accounts: detect on lines with context keywords and mask long digit groups
    if pats.get("bank_account", True):
        ctx = [c.lower() for c in cfg.get("context_keywords", {}).get("bank_account", [])]
        new_lines = []
        for line in red.splitlines():
            llow = line.lower()
            if any(k in llow for k in ctx):
                # Replace digit clusters (e.g., 0011-104097-001 or long num sequences)
                before = line
                line = re.sub(r"\b\d[\d\-\s]{6,}\d\b", mask.get("bank_account", "████ACCOUNT████"), line)
                if line != before:
                    counts["bank_accounts_redacted"] += 1
            new_lines.append(line)
        red = "\n".join(new_lines)

    # Qatar ID: 11-digit patterns near QID context
    if pats.get("qid", True):
        ctx = [c.lower() for c in cfg.get("context_keywords", {}).get("qid", [])]
        new_lines = []
        for line in red.splitlines():
            llow = line.lower()
            if any(k in llow for k in ctx):
                before = line
                line = re.sub(r"\b\d{11}\b", mask.get("qid", "████QID████"), line)
                if line != before:
                    counts["qids_redacted"] += 1
            new_lines.append(line)
        red = "\n".join(new_lines)

    # Payment cards (PAN): Luhn-valid 13–19 digit sequences; context-aware preferred
    if pats.get("pan", True):
        def _mask_pan(m: re.Match) -> str:
            raw = m.group(0)
            digits = re.sub(r"[^0-9]", "", raw)
            return mask.get("pan", "████CARD████") if _luhn_valid(digits) else raw

        red = re.sub(r"\b(?:\d[ -]?){13,19}\b", _mask_pan, red)

    return red, counts


def audit_unmask_event(user_id: str, reason: str, fields: dict, log_path: str = "logs/audit_unmask.jsonl") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "reason": reason,
        "fields": fields,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Non-fatal in demo mode
        pass


def check_financial_integrity(subtotal: float | None, tax: float | None, total: float | None) -> tuple[bool, str]:
    """Return (ok, message) verifying subtotal + tax == total within 0.01."""
    if subtotal is None or tax is None or total is None:
        return False, "Missing values for integrity check"
    calc = (subtotal or 0.0) + (tax or 0.0)
    if abs(calc - (total or 0.0)) < 0.01:
        return True, "Math matches"
    return False, f"Calculated {calc:.2f}, provided total {total:.2f}"
