import json
import os
import io
import zipfile
import tempfile
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="IDP Context Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Context Graph Explorer (3D)")

st.caption("Upload a recon_report (JSONL/JSON) or a ZIP of docs to visualize chains/clusters.")
up = st.file_uploader("Upload recon report or docs (jsonl/json/zip)", type=["jsonl", "json", "zip"]) 


def status_to_onehot(s: str) -> np.ndarray:
    cats = [
        "matched_3_way",
        "matched_po_invoice",
        "matched_invoice_receipt",
        "invoice_only",
        "receipt_only",
        "partial",
    ]
    v = np.zeros(len(cats), dtype=float)
    if s in cats:
        v[cats.index(s)] = 1.0
    return v


def vendor_hash(v: str | None) -> float:
    if not v:
        return 0.0
    return (hash(v) % 1000) / 1000.0


def to_features(rows):
    feats = []
    meta = []
    for r in rows:
        x = [
            float(r.get("po_total") or 0.0),
            float(r.get("invoice_total") or 0.0),
            float(r.get("receipt_amount") or 0.0),
            float(r.get("confidence") or 0.0),
            float(len(r.get("anomalies", []))),
            vendor_hash(r.get("vendor")),
        ]
        x.extend(list(status_to_onehot(r.get("status") or "partial")))
        feats.append(x)
        meta.append({
            "key": r.get("key"),
            "status": r.get("status"),
            "vendor": r.get("vendor"),
            "po_total": r.get("po_total"),
            "invoice_total": r.get("invoice_total"),
            "receipt_amount": r.get("receipt_amount"),
            "anomalies": ", ".join(r.get("anomalies", [])),
        })
    X = np.array(feats, dtype=float)
    # standardize
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mu) / std
    # PCA via SVD with graceful fallback to <3 dims
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    k = int(min(3, U.shape[1] if U.ndim == 2 else 1, S.shape[0]))
    Zk = (U[:, :k] * S[:k]) if k > 0 else np.zeros((X.shape[0], 0))
    if k < 3:
        pad = np.zeros((X.shape[0], 3 - k))
        Z = np.concatenate([Zk, pad], axis=1) if Zk.size else pad
    else:
        Z = Zk
    return Z, meta


def _looks_like_chain(obj: dict) -> bool:
    return any(k in obj for k in ("status", "po_total", "invoice_total", "receipt_amount", "anomalies"))


def _load_chains_from_upload(upload) -> list[dict]:
    name = upload.name.lower()
    data = upload.read()
    # JSONL report
    if name.endswith(".jsonl"):
        rows = []
        for ln in data.decode("utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
        return rows

    # JSON: either a list of chains or list of docs (or single doc)
    if name.endswith(".json"):
        try:
            obj = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return []
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            if _looks_like_chain(obj[0]):
                return obj
            # Treat as docs → chains
            try:
                from agents.reconcile import normalize_input, group_docs
                docs = [normalize_input(o, f"upload:{i}") for i, o in enumerate(obj)]
                return group_docs(docs)
            except Exception:
                return []
        if isinstance(obj, dict):
            # Single chain or single doc
            if _looks_like_chain(obj):
                return [obj]
            try:
                from agents.reconcile import normalize_input, group_docs
                doc = normalize_input(obj, "upload:0")
                return group_docs([doc])
            except Exception:
                return []
        return []

    # ZIP: extract and build chains from contained docs
    if name.endswith(".zip"):
        try:
            from agents.context_graph import iter_docs
            from agents.reconcile import group_docs
            with tempfile.TemporaryDirectory() as td:
                zf = zipfile.ZipFile(io.BytesIO(data))
                zf.extractall(td)
                docs = iter_docs(td)
                return group_docs(docs)
        except Exception:
            return []

    return []


if up is not None:
    rows = _load_chains_from_upload(up)
    if not rows:
        st.error("No chains parsed. Upload a recon_report (.jsonl/.json) or a ZIP of docs.")
    else:
        Z, meta = to_features(rows)
        df = {
            "x": Z[:, 0],
            "y": Z[:, 1],
            "z": Z[:, 2],
            "status": [m["status"] for m in meta],
            "key": [m["key"] for m in meta],
            "vendor": [m["vendor"] for m in meta],
            "anomalies": [m["anomalies"] for m in meta],
        }
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="status", hover_data=["key", "vendor", "anomalies"], opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a report to see clusters. Generate one via Reconcile or agents/context_graph.py.")
