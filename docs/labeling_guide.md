Labeling Guide (Qatar P2P)

Scope
- Goal: High‑value header KIE with bboxes for 4 templates: Al Mirza (invoice/proforma), ORYXI/Manycon (invoice), Qasr Al Abwab (invoice), MTC (job completion/delivery note receipt).
- Output: One JSON per document following data/labeling/schemas/kie_schema.json with values and bounding boxes.

What to Label (per type)
- Invoice (Al Mirza, ORYXI/Manycon, Qasr):
  - invoice_number, invoice_date (optional if unreadable), due_date (if present)
  - vendor, currency (QAR/QR/ر.ق), subtotal, tax (0 if absent), total
  - lpo_ref (PO/LPO reference), payment_received/final_payment (if present)
- Receipt (MTC job completion/delivery note):
  - reference, date, buyer, vendor, lpo_ref, invoice_ref (if present), amount (if present)
- PO (if any):
  - po_number, po_date, buyer, vendor, currency, total_amount, lpo_ref

Bounding Boxes
- Draw tight boxes around the value tokens, not the label text.
  - Example: For “Invoice number: 5836”, box only “5836”. For totals, box the numeric amount “QAR 1,000.00”.
- Coordinates: axis-aligned [x0, y0, x1, y1] in PDF pixel space for the specific page.
- Multi-line values: use a single box covering all characters if visually contiguous; otherwise, provide multiple spans in the field’s spans array.

PII/Governance
- Do not redact in labels. The pipeline masks in UI/logs; labels are stored encrypted internally.
- Mark PII fields (emails, phones, IBAN/SWIFT) only if they are part of a required key (generally they are not for KIE).

Quality Rules
- Dates: prefer ISO yyyy-mm-dd in the JSON value; keep original text in a separate ‘raw’ if helpful.
- Currency/amounts: value as float; preserve currency code in ‘currency’ field.
- Unknown/missing: use null; never guess.
- Cross-check: subtotal + tax ≈ total (±0.01); note any mismatch in ‘notes’.

File Naming & Placement
- Place labeled JSON next to source file under:
  - data/processed/train|val|test/<template>/docname.json
- Keep the original PDF/image alongside for audit if possible.

Review Checklist (per doc)
- Keys present for the doc type; boxes are tight; totals math passes or is noted; LPO/ref links captured; dates normalized.

