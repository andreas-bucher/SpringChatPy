'''
Created on 5 Feb 2026

@author: ante
'''
import fitz  # PyMuPDF
import json, os, sys, logging
import uuid, hashlib
import re
from statistics import median
from pathlib import Path

log = logging.getLogger(__name__)

PDF_PATH = "/mnt/data/Module 1_Quick Reference Guide.pdf"
OUT_JSONL = "/mnt/data/module1_chunks_lossless.jsonl"

MAX_CHARS = 1200
OVERLAP_CHARS = 150

HEADER_SCALE_H1 = 1.35
HEADER_SCALE_H2 = 1.20
TITLE_SCALE     = 1.75


def normalize_text(s: str) -> str:
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_bold_font(font_name: str) -> bool:
    return "bold" in (font_name or "").lower()


def page_body_font_size(page_dict):
    sizes = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if (s.get("text") or "").strip():
                    sz = s.get("size")
                    if isinstance(sz, (int, float)):
                        sizes.append(float(sz))
    return median(sizes) if sizes else 0.0


def block_text_and_style(block):
    parts, sizes = [], []
    bold = False
    for l in block.get("lines", []):
        for s in l.get("spans", []):
            parts.append(s.get("text", ""))
            sz = s.get("size")
            if isinstance(sz, (int, float)):
                sizes.append(float(sz))
            bold |= is_bold_font(s.get("font", ""))
    text = "".join(parts).strip()
    avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
    return text, avg_size, bold


def classify(avg_size, body_size):
    if body_size > 0 and avg_size >= body_size * TITLE_SCALE:
        return "title"
    if body_size > 0 and avg_size >= body_size * HEADER_SCALE_H1:
        return "h1"
    if body_size > 0 and avg_size >= body_size * HEADER_SCALE_H2:
        return "h2"
    return "body"


def split_chunks(text, max_chars, overlap):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def short_hash(text: str) -> str:
    # stable id helper (for dedupe), not used as Qdrant id (we still use UUID)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def extract_all_text(pdf_path, out_jsonl):
    source_file = Path(pdf_path).name
    doc = fitz.open(pdf_path)

    ctx = {"title": None, "h1": None, "h2": None}
    chunk_index = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for page in doc:
            page_dict = page.get_text("dict")
            body_size = page_body_font_size(page_dict)

            blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))  # reading order

            for block_index, b in enumerate(blocks):
                raw, avg_size, bold = block_text_and_style(b)
                text = normalize_text(raw)
                if not text:
                    continue

                level = classify(avg_size, body_size)

                # Update hierarchy context, but DO NOT skip emitting
                if level == "title":
                    ctx["title"] = text
                    ctx["h1"] = None
                    ctx["h2"] = None
                elif level == "h1":
                    ctx["h1"] = text
                    ctx["h2"] = None
                elif level == "h2":
                    ctx["h2"] = text

                section_path = " > ".join([p for p in (ctx["title"], ctx["h1"], ctx["h2"]) if p])

                for part_index, chunk in enumerate(split_chunks(text, MAX_CHARS, OVERLAP_CHARS)):
                    record = {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {
                            "source_file": source_file,
                            "page": page.number + 1,
                            "reading_order": block_index,      # useful replacement for bbox
                            "part_index": part_index,          # which chunk within the block
                            "level": level,
                            "font_size_avg": round(avg_size, 2),
                            "is_bold": bool(bold),
                            "title": ctx["title"],
                            "h1": ctx["h1"],
                            "h2": ctx["h2"],
                            "section_path": section_path or None,
                            "text_hash": short_hash(chunk),    # helpful for dedupe/debug
                            "chunk_index": chunk_index,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_index += 1

    doc.close()


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
        log.info(f"Processing {pdf_file} ...")
        chunks = extract_all_text(pdf_path, out_path)
    log.info("Done.")