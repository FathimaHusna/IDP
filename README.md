# IDP

Demo: https://irlshsv92vdd4rzbbtlq9w.streamlit.app/

IDP Prototype (Local, No Cloud)

Overview
- Minimal, file-based prototype of an Intelligent Document Processing (IDP) pipeline.
- Simulates OCR, classification, entity extraction, validation, scoring, and routing.
- No external services or packages required; uses plain Python 3.

Quick Start
- Ensure Python 3.9+ is available.
- Run: `python3 -m proto.main samples`

Streamlit Apps
- Governance app: `streamlit run apps/governance_app.py`
- 3D Graph viewer: `streamlit run apps/context_graph_app.py`

What It Does
- Reads `.txt` documents from the provided folder.
- Classifies into invoice, purchase order, or contract using simple heuristics.
- Extracts entities per type using regex-based parsers.
- Validates fields (e.g., invoice totals math) and computes a final confidence.
- Prints a JSON result per document (no writes required).

Structure
- `proto/` core modules (orchestrator, ocr, classify, extract, validate, schemas, utils).
- `samples/` a few example documents in `.txt` format.

Next Steps (to reach Azure POC)
- Swap `ocr.py` with Azure Document Intelligence client.
- Replace `classify.py` heuristic with Azure ML endpoint call.
- Replace `extract.py` with Azure OpenAI GPT-4 JSON-mode call.
- Persist to CosmosDB instead of printing.
- Add Durable Functions or a simple queue to orchestrate.

Data Scaffolding (Qatar P2P)
- Registry at `data/registry.yaml` lists public, internal, and synthetic datasets.
- Internal folders: `data/raw/internal/{invoices,pos,receipts}/` and `data/raw/internal/vendor_master.csv`.
- Synthetic folders: `data/synthetic/{pos,invoices,receipts,chains}/`.

Generate Synthetic PO→Invoice→Job Completion Chains (QAR)
- Script: `scripts/synth_qatar_p2p.py`
- Run: `python scripts/synth_qatar_p2p.py --n 100`
- Options: `--out data/synthetic` `--vendor_csv data/raw/internal/vendor_master.csv` `--buyer "ORYXI Maintenance Services (Buyer)"` `--seed 7`
- Outputs:
  - JSON docs in `data/synthetic/pos/*.json`, `invoices/*.json`, `receipts/*.json`
  - Graph edges JSONL in `data/synthetic/chains/edges.jsonl`

Domain Notes
- Currency: QAR; invoices may be bilingual (e.g., "فاتورة ضريبية").
- Links: LPO/PO references connect PO ↔ Invoice; Job Completion Certificate acts as receipt.
- Terms: support `100% upon completion`, `40% advance 60% upon delivery`, `40/55/5 retention`.
- KPIs: math checks, 2-/3-way match, anomaly flags for mismatches.

Render Synthetic JSON to PDFs (for UI uploads)
- Script: `scripts/render_synth_to_pdf.py`
- Run: `python scripts/render_synth_to_pdf.py --src data/synthetic --out data/synthetic_pdf --limit 10`
- Outputs:
  - PDFs under `data/synthetic_pdf/{pos,invoices,receipts}/` which you can upload in the Streamlit Reconcile tab.

Reconciliation Agent (Agentic Demo)
- Script: `agents/reconcile.py`
- Input: Folder with synthetic JSON (from `scripts/synth_qatar_p2p.py`) or real docs (JSON from pipeline, PDFs/TXTs will be OCR’d via local pipeline if available).
- Run:
  - On synthetic: `python agents/reconcile.py data/synthetic --out recon_report.jsonl`
  - On a mixed folder: `python agents/reconcile.py path/to/folder --out recon_report.jsonl`
- Output: JSONL chains with:
  - key (PO/LPO), vendor, po_total, invoice_total, receipt_amount
  - counts (po/invoices/receipts), sources
  - status: matched_3_way | matched_po_invoice | matched_invoice_receipt | invoice_only | receipt_only | partial
  - anomalies: [invoice_math_mismatch, po_vs_invoice_total_mismatch, receipt_vs_invoice_amount_mismatch]
  - confidence: 0–1 (router/anchor-aware)

Benchmarks & Health
- Health check: `python scripts/healthcheck.py`
- Benchmark samples → logs/metrics.json: `python scripts/benchmark_cli.py samples --out logs/metrics.json`

Auto Watcher (optional)
- Continuously process new files in a folder and update metrics:

```bash
python scripts/auto_watch.py samples --interval 10 --audit
# or via Makefile
make watch
```

Docker
- Build: `docker build -t idp-app:latest .`
- Run: `docker run --rm -p 8501:8501 -e IDP_OCR_FAST=1 idp-app:latest`

Env Flags
- `IDP_ENABLE_AGENTIC_TAB=1` show Agentic Runs tab
- `IDP_ENABLE_GRAPH_TAB=0` hide Graph tab
- `IDP_ENABLE_CROSSDOC=0` disable cross-doc agents (faster)
- `IDP_ENABLE_SPACY=1` enable exemplar vectors (slower)
- `IDP_OCR_FAST=1` force minimal OCR on PDFs
- `IDP_AUDIT=1` write agent timing events to `logs/runs/<run_id>.jsonl`
