import streamlit as st
import os
from common.paths import LOGS_DIR, GRAPHS_DIR, ensure_dirs, rel
from proto.governance import enforce_privacy_policy, check_financial_integrity
from proto.classify import classify
from proto.extract import extract
from proto.validate import validate
from proto.orchestrator import score_final
from proto.utils import sha256_bytes
from proto.ocr import ocr_from_bytes
from proto.router import detect_template
from agents.reconcile import normalize_input as recon_normalize_input, group_docs as recon_group_docs

# --- Preflight checks for OCR dependencies ---
def _tesseract_ok() -> bool:
    try:
        import pytesseract  # type: ignore

        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _poppler_note() -> str | None:
    try:
        import pdf2image  # type: ignore

        # Presence of module is a good hint; Poppler availability will be checked at runtime.
        return None
    except Exception:
        return "pdf2image not installed; PDF OCR may be unavailable."

st.set_page_config(page_title="IDP Governance & Compliance", page_icon="🛡️", layout="centered")
st.title("📄 IDP: Governance & Compliance Prototype")

# Sidebar: Environment status
with st.sidebar:
    st.subheader("Environment Status")
    if _tesseract_ok():
        st.success("Tesseract: available")
    else:
        st.error("Tesseract: not found. Install to enable image/PDF OCR.")
        st.caption("Ubuntu: apt-get install tesseract-ocr • macOS: brew install tesseract • Windows: install UB Mannheim build and add to PATH")
    note = _poppler_note()
    if note:
        st.warning(note)
        st.caption("Ubuntu: apt-get install poppler-utils • macOS: brew install poppler • Windows: install Poppler and add to PATH")
    st.divider()
    st.subheader("LLM Automation")
    import os as _os
    prov = _os.environ.get("LLM_PROVIDER", "-")
    default_llm = _os.environ.get("LLM_ENABLED", "0") == "1"
    st.session_state.setdefault("llm_enabled_toggle", default_llm)
    st.checkbox("Enable LLM agents", key="llm_enabled_toggle")
    st.caption(f"Provider: {prov}")
    if st.session_state["llm_enabled_toggle"]:
        # Surface model hints without exposing secrets
        mdl = _os.environ.get("OPENAI_MODEL") or _os.environ.get("AZURE_OPENAI_DEPLOYMENT") or _os.environ.get("LOCAL_LLM_MODEL") or "(model env not set)"
        st.caption(f"Model: {mdl}")
        # Minimal config sanity
        if prov == "openai" and not _os.environ.get("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY not set; LLM calls will fail.")
        if prov == "azure" and not (_os.environ.get("AZURE_OPENAI_ENDPOINT") and _os.environ.get("AZURE_OPENAI_API_KEY") and _os.environ.get("AZURE_OPENAI_DEPLOYMENT")):
            st.warning("Azure OpenAI envs not fully set; LLM calls will fail.")

# Build modes dynamically to keep UI lean by default
enable_agentic = os.environ.get("IDP_ENABLE_AGENTIC_TAB", "0") == "1"
enable_graph = os.environ.get("IDP_ENABLE_GRAPH_TAB", "1") == "1"
enable_manual = os.environ.get("IDP_ENABLE_MANUAL_TAB", "0") == "1"

available_modes = ["Upload document", "Reconcile"]
if enable_agentic:
    available_modes.append("Agentic Runs")
if enable_graph:
    available_modes.append("Graph")
if enable_manual:
    available_modes.append("Manual inputs")
available_modes.append("Metrics")

mode = st.radio("Mode", available_modes, horizontal=True)

if mode == "Upload document":
    up = st.file_uploader(
        "Upload a document (.txt, image, or PDF)",
        type=["txt", "png", "jpg", "jpeg", "pdf"],
        help="Runs local pipeline: OCR (if needed) → classify → extract → validate → governance → score → route",
    )
    fast_mode = st.checkbox("Fast mode (prefer embedded PDF text; skip heavy OCR)", value=True)
    if up is not None:
        raw = up.read()
        # File hygiene: size and mime
        from common.security import sniff_mime, file_size_mb, pdf_page_count_from_bytes, antivirus_scan_bytes
        m = sniff_mime(raw, up.name)
        size_mb = file_size_mb(raw)
        if size_mb > 20:
            st.error(f"File too large ({size_mb:.1f} MB). Limit is 20 MB.")
            st.stop()
        allowed = {"application/pdf", "image/jpeg", "image/png", "text/plain"}
        if m not in allowed:
            st.error(f"File type not allowed: {m}")
            st.stop()
        ok, details = antivirus_scan_bytes(raw, (up.name.split('.')[-1] if '.' in up.name else None))
        if not ok:
            st.error("Antivirus flagged the upload. Rejecting.")
            st.stop()
        if m == "application/pdf":
            pc = pdf_page_count_from_bytes(raw)
            if pc and pc > 30:
                st.error(f"PDF has too many pages ({pc}). Limit is 30.")
                st.stop()
        suffix = (up.name.split(".")[-1].lower() if "." in up.name else None)
        text = ""
        ocr_quality = None
        if suffix in {"txt", "text"}:
            try:
                text = raw.decode("utf-8", errors="ignore")
                ocr_quality = 0.98
            except Exception:
                text = raw.decode("latin-1", errors="ignore")
                ocr_quality = 0.9
        else:
            with st.spinner("Running OCR / text extraction..."):
                if suffix == "pdf" and fast_mode:
                    try:
                        from pypdf import PdfReader  # type: ignore
                        import io
                        reader = PdfReader(io.BytesIO(raw))
                        max_pages = min(len(reader.pages), 5)
                        texts = []
                        for i in range(max_pages):
                            try:
                                texts.append(reader.pages[i].extract_text() or "")
                            except Exception:
                                break
                        pdf_text = "\n".join(texts).strip()
                        ocr_result = {"text": pdf_text, "quality": 0.95 if pdf_text else 0.6}
                    except Exception:
                        # No pypdf: choose fast OCR fallback (1 page, low DPI)
                        import os as _os
                        _os.environ["IDP_OCR_FAST"] = "1"
                        ocr_result = ocr_from_bytes(raw, suffix)
                else:
                    ocr_result = ocr_from_bytes(raw, suffix)
            text = ocr_result.get("text", "")
            ocr_quality = ocr_result.get("quality", 0.6)

        # Pipeline
        doc_id = sha256_bytes(raw)
        with st.spinner("Classifying document..."):
            clf = classify(text)
        doc_type = clf["type"]
        probs = clf["probs"]
        # Template router override
        with st.spinner("Detecting template..."):
            tmpl = detect_template(text)
        template_name = tmpl.get("template", "unknown")
        force_type = tmpl.get("force_type")
        if force_type:
            doc_type = force_type

        with st.spinner("Extracting fields..."):
            payload = extract(doc_type, text)
        with st.spinner("Validating fields..."):
            val = validate(doc_type, payload)

        # Scoring: router-aware boost + anchor bonuses
        type_prob = probs.get(doc_type, 0.0)
        if force_type:
            type_prob = max(type_prob, 0.8)
        base_conf = score_final(type_prob, val.get("completeness", 0.0), val.get("validation_score", 0.0))
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
        final_conf = min(1.0, base_conf + bonus)

        # Per-type routing thresholds
        if doc_type == "receipt":
            route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.6 else "reject")
        else:
            route = "auto_accept" if final_conf >= 0.9 else ("review" if final_conf >= 0.7 else "reject")

        with st.spinner("Applying governance checks..."):
            redacted, stats = enforce_privacy_policy(text)
        fraud_ok, fraud_msg = (True, "Not applicable")
        if doc_type == "invoice":
            fraud_ok, fraud_msg = check_financial_integrity(payload.get("subtotal"), payload.get("tax"), payload.get("total"))

        st.subheader("Document Summary")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Type", doc_type)
        colB.metric("Final confidence", f"{final_conf:.3f}")
        colC.metric("Route", route)
        colD.metric("OCR quality", f"{ocr_quality:.2f}")
        if template_name:
            st.caption(f"Template: {template_name}")

        if not text.strip():
            st.warning("OCR returned no text. Ensure Tesseract (and Poppler for PDFs) is installed, or try a higher-resolution image.")
        st.caption(f"docId: {doc_id}")

        st.subheader("Classification")
        st.json(probs)

        st.subheader("Extraction")
        st.json(payload)

        st.subheader("Validation")
        st.json(val)

        st.subheader("🛡️ Governance")
        st.text("Privacy Policy Output:")
        st.code(redacted)
        st.caption(f"Emails redacted: {stats['emails_redacted']} • Phones redacted: {stats['phones_redacted']}")

        if doc_type == "invoice":
            if fraud_ok:
                st.success("✅ " + fraud_msg)
            else:
                st.error("🚨 FRAUD RISK: " + fraud_msg)
                st.caption("Action: Document flagged for Manual Review.")

        with st.expander("View extracted text"):
            st.text(text)

    else:
        st.info("Upload a .txt file to run the pipeline dynamically.")

elif mode == "Manual inputs":
    st.subheader("1. Extracted Data (Manual)")
    col1, col2, col3 = st.columns(3)
    extracted_subtotal = col1.number_input("Subtotal", value=100.0, step=0.5)
    extracted_tax = col2.number_input("Tax", value=10.0, step=0.5)
    extracted_total = col3.number_input("Total (on Invoice)", value=120.0, step=0.5)

    extracted_text = st.text_area(
        "Extracted Text", "Contact support at admin@enonte.com or 555-0199 for help.", height=160
    )

    st.divider()
    st.subheader("🛡️ Governance Logic Layer")

    # Privacy check
    redacted, stats = enforce_privacy_policy(extracted_text)
    st.text("Privacy Policy Output:")
    st.code(redacted)
    st.caption(f"Emails redacted: {stats['emails_redacted']} • Phones redacted: {stats['phones_redacted']}")

    # Fraud/math check
    ok, message = check_financial_integrity(extracted_subtotal, extracted_tax, extracted_total)
    if ok:
        st.success("✅ " + message)
    else:
        st.error("🚨 FRAUD RISK: " + message)
        st.caption("Action: Document flagged for Manual Review.")

    st.divider()
    st.caption("This demo uses local governance logic; production integrates with the pipeline validators and audit logs.")

elif mode == "Reconcile":
    st.subheader("Reconcile PO ↔ Invoice ↔ Receipt")
    st.caption("Upload a small bundle (JSON/PDF/TXT); we’ll extract and reconcile.")
    ups = st.file_uploader(
        "Upload documents",
        type=["json", "pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        help="You can mix JSON from synthetic generator and PDFs/TXTs."
    )
    if ups:
        docs = []
        for up in ups:
            raw = up.read()
            name = up.name
            suf = (name.split(".")[-1].lower() if "." in name else "")
            try:
                if suf == "json":
                    import json
                    obj = json.loads(raw.decode("utf-8", errors="ignore"))
                else:
                    # If this looks like a synthetic-rendered PDF (inv_X/po_X/rcp_X), try loading ground-truth JSON
                    loaded_from_json = False
                    base = name.rsplit(".", 1)[0]
                    prefix = base.split("_")[0]
                    if prefix in {"po", "inv", "rcp"}:
                        import os, json as _json
                        guess_map = {
                            "po": os.path.join("data", "synthetic", "pos", base + ".json"),
                            "inv": os.path.join("data", "synthetic", "invoices", base + ".json"),
                            "rcp": os.path.join("data", "synthetic", "receipts", base + ".json"),
                        }
                        guess = guess_map.get(prefix)
                        if guess and os.path.exists(guess):
                            with open(guess, "r", encoding="utf-8") as f:
                                obj = _json.load(f)
                                loaded_from_json = True
                    if not loaded_from_json:
                        # Run local pipeline
                        ocr_result = ocr_from_bytes(raw, suf)
                        text = ocr_result.get("text", "")
                        clf = classify(text)
                        doc_type = clf["type"]
                        tmpl = detect_template(text)
                        if tmpl.get("force_type"):
                            doc_type = tmpl["force_type"]
                        payload = extract(doc_type, text)
                        obj = {"type": doc_type, "extraction": payload}
                docs.append(recon_normalize_input(obj, name))
            except Exception as e:
                st.warning(f"Failed to process {name}: {e}")
        try:
            chains = recon_group_docs(docs)
            import json
            st.success(f"Reconciled {len(chains)} chains from {len(docs)} docs")
            # Summary
            from collections import Counter
            cnt = Counter([c.get("status") for c in chains])
            st.write({"status_counts": dict(cnt)})
            # Table view
            view = [
                {
                    "key": c.get("key"),
                    "status": c.get("status"),
                    "confidence": c.get("confidence"),
                    "vendor": c.get("vendor"),
                    "po_total": c.get("po_total"),
                    "invoice_total": c.get("invoice_total"),
                    "receipt_amount": c.get("receipt_amount"),
                    "anomalies": ", ".join(c.get("anomalies", [])),
                }
                for c in chains
            ]
            st.dataframe(view, use_container_width=True)
            # Download JSONL and persist report + graph using helpers
            try:
                ensure_dirs()
                from agents.context_graph import write_recon_report, write_graph_from_chains
                report_path = LOGS_DIR / "recon_report.jsonl"
                write_recon_report(chains, str(report_path))
                st.caption(f"Saved recon report → {rel(report_path)}")

                nodes_path, edges_path, n_nodes, n_edges = write_graph_from_chains(chains, str(GRAPHS_DIR))
                st.caption(f"Graph generated: {n_nodes} nodes, {n_edges} edges → {rel(GRAPHS_DIR / 'nodes.jsonl')}, {rel(GRAPHS_DIR / 'edges.jsonl')}")

                # Offer downloads for graph files
                import io
                with open(nodes_path, "rb") as f:
                    nodes_bytes = f.read()
                with open(edges_path, "rb") as f:
                    edges_bytes = f.read()
                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.download_button("Download nodes.jsonl", nodes_bytes, file_name="nodes.jsonl", mime="application/json")
                with col_g2:
                    st.download_button("Download edges.jsonl", edges_bytes, file_name="edges.jsonl", mime="application/json")
                with col_g3:
                    try:
                        if hasattr(st, "page_link"):
                            st.page_link("apps/context_graph_app.py", label="View 3D Graph", icon=":globe_with_meridians:")
                        else:
                            st.link_button("View 3D Graph", "apps/context_graph_app.py")
                    except Exception:
                        st.info("To view 3D graph: run `streamlit run apps/context_graph_app.py` and upload the recon_report.jsonl.")
            except Exception as ge:
                st.warning(f"Graph build skipped: {ge}")
        except Exception as e:
            st.error(f"Reconciliation failed: {e}")

elif mode == "Agentic Runs":
    from agents.agent_orchestrator import run_on_path, run_folder, run_folder_with_recon
    import os
    import json
    import tempfile

    st.subheader("Agentic Orchestrator")
    st.caption("Run the multi-agent pipeline on a file or a folder. Shows agent logs, decisions, and outputs.")

    tab_file, tab_folder = st.tabs(["Single file", "Folder batch"])

    with tab_file:
        up = st.file_uploader(
            "Upload a document (.txt, image, or PDF)",
            type=["txt", "png", "jpg", "jpeg", "pdf"],
            help="We will save it to a temp folder and run the orchestrator.",
            key="agentic_single_file",
        )
        if up is not None:
            try:
                ensure_dirs()
                run_dir = LOGS_DIR / "agentic_runs"
                run_dir.mkdir(parents=True, exist_ok=True)
                # Use storage adapter to save
                from common.storage import Storage
                raw = up.read()
                key = f"agentic_runs/{up.name}"
                tmp_path = Storage().save_bytes(key, raw)
                with st.spinner("Processing with agent orchestrator..."):
                    res = run_on_path(tmp_path, use_llm=bool(st.session_state.get("llm_enabled_toggle")))
                st.success(f"Processed: {up.name}")
                # Summary metrics
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Type", res.get("type"))
                colB.metric("Final confidence", f"{res.get('final_confidence', 0.0):.3f}")
                colC.metric("Route", res.get("route"))
                colD.metric("OCR quality", f"{res.get('ocr_quality', 0.0):.2f}")
                if res.get("template"):
                    st.caption(f"Template: {res['template']}")

                st.subheader("Agent Logs")
                st.code("\n".join(res.get("agent_logs", [])) or "(no logs)")

                st.subheader("Classification")
                st.json(res.get("class_probs", {}))

                st.subheader("Extraction")
                st.json(res.get("extraction", {}))

                st.subheader("Validation")
                st.json(res.get("validation", {}))

                st.subheader("🛡️ Governance")
                st.json(res.get("governance", {}))

                rc = res.get("reason_codes") or []
                if rc:
                    st.subheader("Reason Codes")
                    st.write(", ".join(rc))

                # Planner trace (ReAct-style)
                rt = res.get("react_trace") or []
                if rt:
                    st.subheader("Planner Trace")
                    st.json(rt)

                # LLM banner (success/failure/unused)
                try:
                    llm_on = bool(st.session_state.get("llm_enabled_toggle"))
                    logs = res.get("agent_logs", []) or []
                    llm_errs = [l for l in logs if "llm_" in l and l.strip().startswith("✗")]
                    llm_ok = any("✓ llm_extract" in l for l in logs) or any(
                        isinstance(x, dict) and str(x.get("thought", "")).lower().startswith("llm suggests") for x in rt
                    )
                    if llm_errs:
                        st.error(f"LLM errors: {llm_errs[-1]}")
                    elif llm_on and llm_ok:
                        st.success("LLM assistance applied for this run.")
                    elif llm_on and not llm_ok:
                        st.info("LLM enabled but not invoked (document was confident/consistent).")
                except Exception:
                    pass

                # Agent timings
                ev = res.get("agent_events") or []
                if ev:
                    st.subheader("Agent Timings")
                    st.dataframe(ev, use_container_width=True)

                # Feedback (HITL)
                st.subheader("Feedback")
                fb = st.radio("Decision", ["approve", "reject", "needs_review"], horizontal=True)
                corr = st.text_area("Corrections (JSON optional)", placeholder='{"invoice_number": "INV-123"}')
                if st.button("Submit Feedback"):
                    try:
                        ensure_dirs()
                        import json as _json
                        from common.db import insert_feedback
                        payload = {
                            "file": res.get("file"),
                            "docId": res.get("docId"),
                            "type": res.get("type"),
                            "route": res.get("route"),
                            "final_confidence": res.get("final_confidence"),
                            "reason_codes": res.get("reason_codes"),
                            "decision": fb,
                            "corrections": _json.loads(corr) if corr.strip() else None,
                        }
                        with open(LOGS_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
                            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
                        # Also persist to sqlite
                        insert_feedback(payload)
                        st.success("Feedback recorded")
                    except Exception as e:
                        st.error(f"Failed to record feedback: {e}")
            except Exception as e:
                st.error(f"Agentic run failed: {e}")

    with tab_folder:
        folder = st.text_input("Folder path", value="samples", help="Run orchestrator on all files in this folder")
        if st.button("Run on folder", type="primary"):
            try:
                res_list, chains = run_folder_with_recon(folder, use_llm=bool(st.session_state.get("llm_enabled_toggle")))
                st.success(f"Processed {len(res_list)} files from {folder}")
                # Table view
                rows = [
                    {
                        "file": r.get("file"),
                        "type": r.get("type"),
                        "confidence": r.get("final_confidence"),
                        "route": r.get("route"),
                        "template": r.get("template"),
                        "recon_status": (r.get("reconciliation") or {}).get("status"),
                        "recon_anomalies": ", ".join((r.get("reconciliation") or {}).get("anomalies", [])),
                        "error": r.get("error"),
                    }
                    for r in res_list
                ]
                st.dataframe(rows, use_container_width=True)

                # Download JSONL
                buf = "\n".join([json.dumps(r, ensure_ascii=False) for r in res_list]).encode("utf-8")
                st.download_button("Download agentic_results.jsonl", buf, file_name="agentic_results.jsonl", mime="application/json")

                # Optional: expanders to view logs per file
                with st.expander("Show agent logs per file"):
                    for r in res_list:
                        if r.get("agent_logs"):
                            st.markdown(f"**{r.get('file')}**")
                            st.code("\n".join(r["agent_logs"]))

                # Cross-doc matches summary and report
                try:
                    from collections import Counter
                    st.subheader("Reconciliation Summary (batch)")
                    cnt = Counter([c.get("status") for c in chains])
                    st.write({"status_counts": dict(cnt)})
                    # Save recon_report and build graph files via helpers
                    ensure_dirs()
                    from agents.context_graph import write_recon_report, write_graph_from_chains
                    report_path = LOGS_DIR / "recon_report.jsonl"
                    write_recon_report(chains, str(report_path))
                    st.caption(f"Saved recon report → {rel(report_path)}")

                    nodes_path, edges_path, n_nodes, n_edges = write_graph_from_chains(chains, str(GRAPHS_DIR))
                    st.caption(f"Graph generated: {n_nodes} nodes, {n_edges} edges → {rel(GRAPHS_DIR / 'nodes.jsonl')}, {rel(GRAPHS_DIR / 'edges.jsonl')}")

                    # Download buttons + page link
                    import io
                    with open(nodes_path, "rb") as f:
                        nodes_bytes = f.read()
                    with open(edges_path, "rb") as f:
                        edges_bytes = f.read()
                    col_g1, col_g2, col_g3 = st.columns(3)
                    with col_g1:
                        st.download_button("Download nodes.jsonl", nodes_bytes, file_name="nodes.jsonl", mime="application/json")
                    with col_g2:
                        st.download_button("Download edges.jsonl", edges_bytes, file_name="edges.jsonl", mime="application/json")
                    with col_g3:
                        try:
                            if hasattr(st, "page_link"):
                                st.page_link("apps/context_graph_app.py", label="View 3D Graph", icon=":globe_with_meridians:")
                            else:
                                st.link_button("View 3D Graph", "apps/context_graph_app.py")
                        except Exception:
                            st.info("To view 3D graph: run `streamlit run apps/context_graph_app.py` and upload the recon_report.jsonl.")
                except Exception as ge:
                    st.warning(f"Reconciliation summary/graph step skipped: {ge}")
            except Exception as e:
                st.error(f"Folder run failed: {e}")

elif mode == "Graph":
    import json
    import os
    import numpy as np
    import plotly.express as px

    st.subheader("Context Graph (Inline)")
    st.caption("Load the latest report or upload one. Filter by status/anomalies and view clusters.")

    # Load from disk or upload
    col1, col2 = st.columns(2)
    with col1:
        use_disk = st.checkbox("Load latest logs/recon_report.jsonl", value=True)
    with col2:
        up = st.file_uploader("Or upload recon report (jsonl/json)", type=["jsonl", "json"], key="graph_tab_upload")

    rows: list[dict] = []
    if use_disk and (LOGS_DIR / "recon_report.jsonl").exists():
        try:
            with open(LOGS_DIR / "recon_report.jsonl", "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
        except Exception as e:
            st.warning(f"Failed to read logs/recon_report.jsonl: {e}")
    if not rows and up is not None:
        try:
            if up.name.lower().endswith(".jsonl"):
                for ln in up.read().decode("utf-8", errors="ignore").splitlines():
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
            else:
                obj = json.loads(up.read().decode("utf-8", errors="ignore"))
                if isinstance(obj, list):
                    rows = obj
                elif isinstance(obj, dict):
                    rows = [obj]
        except Exception as e:
            st.error(f"Failed to parse upload: {e}")

    if not rows:
        st.info("No chains loaded. Run Reconcile, or upload a report.")
    else:
        # Filters
        statuses = sorted({r.get("status") for r in rows if r.get("status")})
        anomalies_all = sorted({a for r in rows for a in (r.get("anomalies") or [])})
        f_status = st.multiselect("Status filter", options=statuses, default=statuses)
        f_anoms = st.multiselect("Anomalies filter", options=anomalies_all, default=anomalies_all)

        def _filter(r):
            ok_s = (r.get("status") in f_status) if f_status else True
            anoms = r.get("anomalies") or []
            ok_a = any(a in f_anoms for a in anoms) if f_anoms else True
            return ok_s and ok_a

        rows_f = [r for r in rows if _filter(r)]
        st.write(f"Showing {len(rows_f)} of {len(rows)} chains")

        # 3D projection similar to context_graph_app
        def _vendor_hash(v: str | None) -> float:
            if not v:
                return 0.0
            return (hash(v) % 1000) / 1000.0

        feats = []
        meta = []
        for r in rows_f:
            x = [
                float(r.get("po_total") or 0.0),
                float(r.get("invoice_total") or 0.0),
                float(r.get("receipt_amount") or 0.0),
                float(r.get("confidence") or 0.0),
                float(len(r.get("anomalies", []))),
                _vendor_hash(r.get("vendor")),
            ]
            # status one-hot compact
            sts = {s: i for i, s in enumerate(["matched_3_way","matched_po_invoice","matched_invoice_receipt","invoice_only","receipt_only","partial"])}
            v = [0.0]*len(sts)
            s = r.get("status")
            if s in sts:
                v[sts[s]] = 1.0
            x.extend(v)
            feats.append(x)
            meta.append(r)
        import numpy as np  # local alias
        X = np.array(feats, dtype=float)
        if X.size == 0:
            st.info("No rows after filtering.")
        else:
            mu = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            Xz = (X - mu) / std
            U, S, _ = np.linalg.svd(Xz, full_matrices=False)
            k = int(min(3, U.shape[1] if U.ndim == 2 else 1, S.shape[0]))
            Zk = (U[:, :k] * S[:k]) if k > 0 else np.zeros((X.shape[0], 0))
            if k < 3:
                pad = np.zeros((X.shape[0], 3 - k))
                Z = np.concatenate([Zk, pad], axis=1) if Zk.size else pad
            else:
                Z = Zk
            df = {
                "x": Z[:, 0],
                "y": Z[:, 1],
                "z": Z[:, 2],
                "status": [m.get("status") for m in meta],
                "key": [m.get("key") for m in meta],
                "vendor": [m.get("vendor") for m in meta],
                "anomalies": [", ".join(m.get("anomalies", [])) for m in meta],
            }
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="status", hover_data=["key", "vendor", "anomalies"], opacity=0.85)
            st.plotly_chart(fig, use_container_width=True)

        # Drill-down table
        st.subheader("Chains")
        st.dataframe([
            {
                "key": r.get("key"),
                "status": r.get("status"),
                "confidence": r.get("confidence"),
                "vendor": r.get("vendor"),
                "po_total": r.get("po_total"),
                "invoice_total": r.get("invoice_total"),
                "receipt_amount": r.get("receipt_amount"),
                "anomalies": ", ".join(r.get("anomalies", [])),
            }
            for r in rows_f
        ], use_container_width=True)

elif mode == "Metrics":
    import json
    import glob
    st.subheader("Operational Metrics")
    col1, col2 = st.columns(2)
    with col1:
        if (LOGS_DIR / "metrics.json").exists():
            m = json.loads((LOGS_DIR / "metrics.json").read_text(encoding="utf-8"))
            st.write("Routes", m.get("routes", {}))
            st.write("Types", m.get("types", {}))
            st.write("Means", m.get("means", {}))
            st.write("Reconciliation Status", (m.get("reconciliation", {}) or {}).get("status_counts", {}))
            st.write("Reason Codes", m.get("reason_codes", {}))
            # Acceptance gates check
            try:
                import yaml, json as _json
                from pathlib import Path as _Path
                gates = (yaml.safe_load((_Path("config/acceptance.yaml")).read_text(encoding="utf-8")) or {}).get("gates", {})
                errs = []
                counts = m.get("counts", {})
                types = m.get("types", {})
                means = m.get("means", {})
                reasons = m.get("reason_codes", {})
                if counts.get("documents", 0) < gates.get("min_documents", 1):
                    errs.append("min_documents failed")
                if counts.get("errors", 0) > gates.get("max_errors", 0):
                    errs.append("max_errors failed")
                if means.get("ocr_quality", 1.0) < gates.get("min_mean_ocr_quality", 0.0):
                    errs.append("min_mean_ocr_quality failed")
                if means.get("final_confidence", 1.0) < gates.get("min_mean_final_confidence", 0.0):
                    errs.append("min_mean_final_confidence failed")
                for t, cons in (gates.get("expected_types", {}) or {}).items():
                    if (types.get(t, 0) or 0) < (cons.get("min") or 0):
                        errs.append(f"expected_types {t} min failed")
                for rc, cnt in (reasons or {}).items():
                    if rc in (gates.get("forbid_reason_codes", []) or []) and cnt > 0:
                        errs.append(f"forbidden reason: {rc}")
                if errs:
                    st.error({"acceptance_errors": errs})
                else:
                    st.success("Acceptance gates passed")
            except Exception as e:
                st.warning(f"Acceptance check failed to run: {e}")
        else:
            st.info("No metrics.json found. Run the benchmark or Agentic batch.")
        # History chart from metrics-*.json
        import glob, os
        files = sorted(glob.glob(str(LOGS_DIR / "metrics-*.json")))
        if files:
            import pandas as pd
            rows = []
            for fp in files[-50:]:
                try:
                    mm = json.loads(open(fp, "r", encoding="utf-8").read())
                    # parse ts from filename metrics-YYYYmmddHHMMSS.json
                    ts_str = os.path.basename(fp).split("-")[1].split(".")[0]
                    from datetime import datetime as _dt
                    ts = _dt.strptime(ts_str, "%Y%m%d%H%M%S")
                    rows.append({
                        "ts": ts,
                        "final_conf_mean": (mm.get("means", {}) or {}).get("final_confidence", 0.0),
                        "ocr_q_mean": (mm.get("means", {}) or {}).get("ocr_quality", 0.0),
                    })
                except Exception:
                    pass
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows).set_index("ts").sort_index()
                st.subheader("History: Means Over Time")
                st.line_chart(df)
    with col2:
        st.write("Recent Agent Runs")
        runs = sorted(glob.glob(str(LOGS_DIR / "runs" / "*.jsonl")))
        if not runs:
            st.caption("No run logs.")
        else:
            sel = st.selectbox("Select run", runs, index=len(runs) - 1)
            evs = []
            with open(sel, "r", encoding="utf-8") as f:
                for ln in f:
                    try:
                        evs.append(json.loads(ln))
                    except Exception:
                        pass
            st.dataframe(evs, use_container_width=True)
            # Simple P95 latency per agent in this run
            from collections import defaultdict
            import numpy as _np
            ag = defaultdict(list)
            for e in evs:
                if e.get("status") == "ok":
                    ag[e.get("agent")].append(int(e.get("duration_ms") or 0))
            rows = []
            for agent, durs in ag.items():
                if durs:
                    p95 = float(_np.percentile(durs, 95))
                else:
                    p95 = 0.0
                rows.append({"agent": agent, "p95_ms": int(p95), "n": len(durs)})
            if rows:

                st.subheader("Latency P95 (this run)")
                st.dataframe(rows, use_container_width=True)
        # Fleet-wide P95 from SQLite
        try:
            import sqlite3
            con = sqlite3.connect(str(LOGS_DIR / "app.db"))
            cur = con.cursor()
            cur.execute("SELECT agent, duration_ms FROM agent_events WHERE status='ok'")
            from collections import defaultdict
            import numpy as _np
            agg = defaultdict(list)
            for a, d in cur.fetchall():
                try:
                    agg[a].append(int(d))
                except Exception:
                    pass
            rows2 = []
            for a, ds in agg.items():
                if not ds:
                    continue
                rows2.append({"agent": a, "p95_ms": int(float(_np.percentile(ds, 95))), "n": len(ds)})
            if rows2:
                st.subheader("Fleet Latency P95 (all runs)")
                st.dataframe(rows2, use_container_width=True)
        except Exception as e:
            st.caption(f"Latency aggregate unavailable: {e}")
