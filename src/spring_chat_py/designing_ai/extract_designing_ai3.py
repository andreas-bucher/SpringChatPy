'''
Created on 5 Feb 2026

@author: ante
'''
import fitz  # PyMuPDF
import json, os, sys, logging
import uuid, hashlib
import re, requests
from statistics import median
from pathlib import Path
from typing import List, Dict, Any
from spring_chat_py.embeddings import embed_chunks

log = logging.getLogger(__name__)


def short_hash(text: str) -> str:
    # stable id helper (for dedupe), not used as Qdrant id (we still use UUID)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def extract_all_text(pdf_path) -> List[Dict[str, Any]]:
    source_file = Path(pdf_path).name
    doc = fitz.open(pdf_path)

    records: List[Dict[str, Any]] = []

    for page in doc:
        text = page.get_text("text")
        #log.info(text)
        record = {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "metadata": {
                        "source_file": source_file,
                        "page": page.number + 1,
                    },
                }
        #f.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
    doc.close()
    return  records

OLLAMA_URL = "http://localhost:11434/api/embed"


def embed_chunks(chunks: List[Dict[str, Any]], model: str = "bge-m3", batch_size: int = 64) -> None:
    log.debug("embed_chunks")
    for b in range(0, len(chunks), batch_size):
        batch = chunks[b : b + batch_size]
        inputs = [c["text"] for c in batch]
        payload = {"model": model, "input": inputs}
        #log.debug("payload: %s", payload)
        #log.debug("")
        # Embeddings API: input can be a list of strings
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        log.debug("response: %s", resp)
        #log.debug("body: %s", resp.json())
        log.debug("")
        # resp.data aligns with inputs order
        for item, emb in zip(batch, resp.json()["embeddings"]):
            item["embedding_model"] = model
            item["embeddings"] = emb
    


def create_extracts(scan_dir: str, chunk_dir: str):
    if not os.path.isdir(scan_dir):
        log.info(f"Scan directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(chunk_dir, exist_ok=True)
    pdf_files = sorted(
        f for f in os.listdir(scan_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        log.info(f"No PDF files found in {scan_dir}", file=sys.stderr)
        sys.exit(1)
        
    for pdf_file in pdf_files:
        pdf_path = os.path.join(scan_dir, pdf_file)
        out_path = os.path.join(
            chunk_dir,
            os.path.splitext(pdf_file)[0] + ".jsonl"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            log.info(f"Processing {pdf_file} ...")
            records = extract_all_text(pdf_path)
            log.debug("records length: %s", len(records))
            embed_chunks(records)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    log.info("Done.")