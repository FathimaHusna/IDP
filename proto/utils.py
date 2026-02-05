import hashlib
import re
from datetime import datetime


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_date(s: str) -> str | None:
    s = s.strip()
    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d-%b-%Y",      # 12-OCT-2023
        "%d-%b-%y",
        "%B %d, %Y",     # January 31, 2026
        "%d.%m.%Y",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            continue
    return None


def as_float(s: str) -> float | None:
    if s is None:
        return None
    txt = s.strip()
    # Remove leading currency codes/symbols (e.g., QAR, QR, USD, ر.ق, £, $, €)
    txt = re.sub(r"^(?:[A-Z]{2,3}|ر\.ق)\s*", "", txt, flags=re.IGNORECASE)
    # Grab the first numeric token (supports commas and decimals)
    m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", txt)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except ValueError:
        return None


def currency_from_symbol_or_code(s: str | None) -> str | None:
    if not s:
        return None
    if "£" in s or s.strip().upper() == "GBP":
        return "GBP"
    if "$" in s or s.strip().upper() == "USD":
        return "USD"
    if "€" in s or s.strip().upper() == "EUR":
        return "EUR"
    if "ر.ق" in s or s.strip().upper().find("QAR") != -1 or re.search(r"\bQR\b", s.strip().upper()):
        return "QAR"
    # Whitelist common codes to avoid false positives like 'AND'
    allowed = {"QAR", "USD", "EUR", "GBP", "AUD", "CAD", "INR", "SGD", "AED", "SAR"}
    m = re.search(r"\b([A-Z]{3})\b", s.strip().upper())
    code = m.group(1) if m else None
    return code if code in allowed else None
