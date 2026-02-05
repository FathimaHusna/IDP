INVOICE_SCHEMA = {
    "required": [
        "invoice_number",
        # "invoice_date",  # optional due to OCR variance
        "vendor",
        "currency",
        "subtotal",
        "tax",
        "total",
        # line_items optional in noisy OCR layouts
    ]
}

PO_SCHEMA = {
    "required": [
        "po_number",
        "po_date",
        "buyer",
        "vendor",
        "currency",
        "total_amount",
        # line_items optional
    ]
}

CONTRACT_SCHEMA = {
    "required": [
        "party_a",
        "party_b",
        "effective_date",
        "term_months",
        "governing_law",
    ]
}

RECEIPT_SCHEMA = {
    "required": [
        "reference",
        "date",
        "buyer",
        "vendor",
        # amount often omitted on job completion/delivery notes
    ]
}
