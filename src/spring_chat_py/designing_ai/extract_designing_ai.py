'''
Created on 5 Feb 2026

@author: ante
'''

import logging
import fitz  # PyMuPDF
import re, os, sys
import json
import uuid
from statistics import median
from pathlib import Path


log = logging.getLogger(__name__)

#PDF_PATH = "/mnt/data/Module 1_Quick Reference Guide.pdf"
#OUT_JSONL = "/mnt/data/module1_chunks.jsonl"

# --- Chunking ---
MAX_CHARS = 1200        # target chunk size
OVERLAP_CHARS = 150     # overlap between chunks (only within the same block)

# --- Header detection heuristics ---
HEADER_SCALE_H1 = 1.35
HEADER_SCALE_H2 = 1.20
TITLE_SCALE     = 1.75
TOP_BOTTOM_MARGIN_PX = 45


def is_bold_font(font_name: str) -> bool:
    f = (font_name or "").lower()
    return "bold" in f or "black" in f or "demi" in f


def looks_like_heading(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    words = t.split()
    if len(words) < 1 or len(words) > 18:
        return False
    if t.endswith("."):
        return False
    return True


def normalize_text(s: str) -> str:
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)            # join hyphenated breaks
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)            # single newlines -> spaces
    s = re.sub(r"[ \t]+", " ", s)                     # collapse spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)                  # collapse blank lines
    return s.strip()


def page_body_font_size(page_dict: dict) -> float:
    sizes = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                sz = span.get("size")
                if txt and isinstance(sz, (int, float)) and sz >= 6:
                    sizes.append(float(sz))
    return median(sizes) if sizes else 0.0


def block_text_and_style(block: dict):
    parts, sizes = [], []
    any_bold = False

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            txt = span.get("text", "")
            if txt:
                parts.append(txt)
            sz = span.get("size")
            if isinstance(sz, (int, float)):
                sizes.append(float(sz))
            any_bold |= is_bold_font(span.get("font", ""))

    text = "".join(parts).strip()
    avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
    return text, avg_size, any_bold


def classify_level(avg_size: float, body_size: float) -> str:
    if body_size <= 0:
        return "body"
    if avg_size >= body_size * TITLE_SCALE:
        return "title"
    if avg_size >= body_size * HEADER_SCALE_H1:
        return "h1"
    if avg_size >= body_size * HEADER_SCALE_H2:
        return "h2"
    return "body"


def is_header_footer_bbox(bbox, page_height: float) -> bool:
    y0, y1 = bbox[1], bbox[3]
    return (y0 < TOP_BOTTOM_MARGIN_PX) or (y1 > page_height - TOP_BOTTOM_MARGIN_PX)


def split_into_chunks(text: str, max_chars: int, overlap: int):
    """
    Simple character-based chunking with overlap, tries to split on paragraph boundaries first.
    """
    t = text.strip()
    if len(t) <= max_chars:
        return [t]

    paras = re.split(r"\n\s*\n", t)
    chunks = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        p = p.strip()
        if not p:
            continue

        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            # flush what we have
            flush_buf()
            # if paragraph itself too big, hard-split it
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunk = p[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= len(p):
                        break
                    start = max(0, end - overlap)
            else:
                buf = p

    flush_buf()

    # add overlap between chunks (only if overlap > 0 and multiple chunks)
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = overlapped[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append((tail + ch).strip())
        return overlapped

    return chunks


def pdf_to_qdrant_jsonl(pdf_path: str, out_jsonl: str):
    pdf_path = str(pdf_path)
    out_jsonl = str(out_jsonl)

    source_file = Path(pdf_path).name
    running_chunk_index = 0

    doc = fitz.open(pdf_path)

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        ctx = {"title": None, "h1": None, "h2": None}

        for page in doc:
            page_dict = page.get_text("dict")
            body_size = page_body_font_size(page_dict)
            page_h = float(page.rect.height)

            blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))  # top-to-bottom

            for b in blocks:
                raw_text, avg_size, any_bold = block_text_and_style(b)
                text = normalize_text(raw_text)
                if not text:
                    continue

                bbox = b.get("bbox", None)

                # Drop obvious header/footer noise unless it truly looks like a heading
                if bbox and is_header_footer_bbox(bbox, page_h) and not looks_like_heading(text):
                    continue

                level = classify_level(avg_size, body_size)
                is_heading = level in ("title", "h1", "h2") and (any_bold or looks_like_heading(text))

                if is_heading:
                    if level == "title":
                        ctx["title"] = text
                        ctx["h1"] = None
                        ctx["h2"] = None
                    elif level == "h1":
                        ctx["h1"] = text
                        ctx["h2"] = None
                    elif level == "h2":
                        ctx["h2"] = text
                    continue  # don't emit headings as body chunks (usually best for RAG)
                
                # Emit body text chunks with current header context
                for chunk_text in split_into_chunks(text, MAX_CHARS, OVERLAP_CHARS):
                    rec = {
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "page": page.number + 1,
                            "bbox": bbox,
                            "title": ctx["title"],
                            "h1": ctx["h1"],
                            "h2": ctx["h2"],
                            "chunk_index": running_chunk_index,
                        },
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    running_chunk_index += 1

    doc.close()
    return out_jsonl


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
        chunks = pdf_to_qdrant_jsonl(pdf_path, out_path)
        
        #if not chunks:
        #    log.info(f"  ⚠️  No text found in {pdf_file}, skipping")
        #    continue

        #with open(out_path, "w", encoding="utf-8") as f:
        #    for row in chunks:
        #        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        #log.info(f"  ✔ Wrote {len(chunks)} chunks → {out_path}")
    log.info("Done.")