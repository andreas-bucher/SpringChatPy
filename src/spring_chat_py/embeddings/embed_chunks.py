'''
Created on 30 Jan 2026

@author: ante
'''
#!/usr/bin/env python3
"""
pdf_embed.py
Minimal: PyMuPDF -> text chunks -> OpenAI embeddings -> JSONL

Usage:
  export OPENAI_API_KEY="..."
  pip install pymupdf openai
  python pdf_embed.py input.pdf output.jsonl
"""

import os, sys, json, logging
from typing import List, Dict, Any
import requests
from spring_chat_py.embeddings.chunks_extractor import extract_chunks_from_pdf


log = logging.getLogger(__name__)

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
    

def create_embeddings(scan_dir: str, embedding_dir: str):
    if not os.path.isdir(scan_dir):
        print(f"Scan directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(embedding_dir, exist_ok=True)
    pdf_files = [
        f for f in os.listdir(scan_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in {scan_dir}", file=sys.stderr)
        sys.exit(1)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(scan_dir, pdf_file)
        out_path = os.path.join(
            embedding_dir,
            os.path.splitext(pdf_file)[0] + ".jsonl"
        )
        print(f"Processing {pdf_file} ...")
        chunks = extract_chunks_from_pdf(pdf_path)
        if not chunks:
            print(f"  ⚠️  No text found in {pdf_file}, skipping")
            continue
        ##################################################################
        # ------------> commented out as this request like costs money
        #               only activate when really needed
        embed_chunks(chunks)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in chunks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  ✔ Wrote {len(chunks)} chunks → {out_path}")
    print("Done.")
    

