#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .reconcile import read_doc, group_docs


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def node_id(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def iter_docs(folder: str) -> List[Dict[str, Any]]:
    files: List[str] = []
    for root, _, fns in os.walk(folder):
        for fn in fns:
            if fn.lower().endswith((".json", ".pdf", ".png", ".jpg", ".jpeg", ".txt")):
                files.append(os.path.join(root, fn))
    docs: List[Dict[str, Any]] = []
    for f in sorted(files):
        d = read_doc(f)
        if d:
            docs.append(d)
    return docs


def build_graph(chains: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(id_: str, label: str, props: Dict[str, Any]) -> None:
        nodes.setdefault(id_, {"id": id_, "label": label, **props})

    def add_edge(src: str, dst: str, rel: str, props: Dict[str, Any] | None = None) -> None:
        edges.append({"src": src, "dst": dst, "rel": rel, **(props or {})})

    for c in chains:
        key = c.get("key") or _hash_id(str(c))
        chain_id = node_id("CHAIN", key)
        add_node(chain_id, "Chain", {"status": c.get("status"), "confidence": c.get("confidence")})

        vendor = c.get("vendor")
        if vendor:
            vend_id = node_id("VENDOR", vendor)
            add_node(vend_id, "Vendor", {"name": vendor})
            add_edge(chain_id, vend_id, "VENDOR")

        po_no = c.get("po_number")
        if po_no:
            po_id = node_id("PO", po_no)
            add_node(po_id, "PO", {"po_number": po_no, "total": c.get("po_total")})
            add_edge(chain_id, po_id, "HAS_PO")

        inv_ids = c.get("invoice_numbers", [])
        for inv in inv_ids:
            if not inv:
                continue
            inv_id = node_id("INV", inv)
            add_node(inv_id, "Invoice", {"invoice_number": inv})
            add_edge(chain_id, inv_id, "HAS_INVOICE", {"total": c.get("invoice_total")})

        r_refs = c.get("receipt_refs", [])
        for r in r_refs:
            if not r:
                continue
            r_id = node_id("RCP", r)
            add_node(r_id, "Receipt", {"reference": r})
            add_edge(chain_id, r_id, "HAS_RECEIPT", {"amount": c.get("receipt_amount")})

        # Anomalies as nodes for easy visualization
        for an in c.get("anomalies", []):
            an_id = node_id("ANOM", an)
            add_node(an_id, "Anomaly", {"name": an})
            add_edge(chain_id, an_id, "FLAG")

        # Sources (documents) assigned to chain
        for s in c.get("sources", []) or []:
            doc_key = node_id("DOC", _hash_id(s))
            add_node(doc_key, "Document", {"source": s})
            add_edge(chain_id, doc_key, "HAS_DOC")

    return list(nodes.values()), edges


def save_jsonl(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_graph_from_chains(chains: List[Dict[str, Any]], out_dir: str = "graphs") -> tuple[str, str, int, int]:
    """Build graph and write nodes/edges.jsonl under out_dir. Returns (nodes_path, edges_path, n_nodes, n_edges)."""
    from pathlib import Path as _Path

    nodes, edges = build_graph(chains)
    d = _Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    nodes_path = str(d / "nodes.jsonl")
    edges_path = str(d / "edges.jsonl")
    save_jsonl(nodes, nodes_path)
    save_jsonl(edges, edges_path)
    return nodes_path, edges_path, len(nodes), len(edges)


def write_recon_report(chains: List[Dict[str, Any]], out_path: str) -> None:
    """Write reconciliation chains JSONL to out_path."""
    save_jsonl(chains, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build context graph from a folder or recon_report.jsonl")
    ap.add_argument("path", help="Folder with docs or recon_report.jsonl")
    ap.add_argument("--nodes_out", default="graphs/nodes.jsonl")
    ap.add_argument("--edges_out", default="graphs/edges.jsonl")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        chains: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chains.append(json.loads(line))
    else:
        docs = iter_docs(str(p))
        chains = group_docs(docs)

    nodes, edges = build_graph(chains)

    Path(args.nodes_out).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(nodes, args.nodes_out)
    save_jsonl(edges, args.edges_out)
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges → {args.nodes_out}, {args.edges_out}")


if __name__ == "__main__":
    main()
