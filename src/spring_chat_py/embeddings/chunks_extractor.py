'''
Created on 30 Jan 2026

@author: ante
'''
from __future__ import annotations

import os, sys, re, json
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF


# -------------------------
# Helpers
# -------------------------

_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS_RE = re.compile(r"[ \t]+")


def _dehyphenate(text: str) -> str:
    # Join hyphenated line breaks: "certifi-\ncate" -> "certificate"
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def _normalize(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = _dehyphenate(text)
    text = text.replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # collapse excessive newlines, but keep paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_header_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Very common noise patterns
    if re.fullmatch(r"page\s*\d+(\s*/\s*\d+)?", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def _bbox_intersects(b1: Tuple[float, float, float, float],
                     b2: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = b1
    a0, b0, a1, b1_ = b2
    return not (x1 <= a0 or a1 <= x0 or y1 <= b0 or b1_ <= y0)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _iter_lines_with_style(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Returns lines in (approx) reading order with minimal style info:
    - text
    - bbox (x0,y0,x1,y1)
    - avg_font_size (float)
    """
    d = page.get_text("dict")
    lines_out: List[Dict[str, Any]] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue

            # Weighted average font size by span length
            sizes = []
            weights = []
            for s in spans:
                t = s.get("text", "")
                if not t:
                    continue
                sizes.append(float(s.get("size", 0.0)))
                weights.append(len(t))
            avg_size = 0.0
            if sizes and weights:
                avg_size = sum(sz * w for sz, w in zip(sizes, weights)) / max(1, sum(weights))

            lines_out.append(
                {
                    "text": text,
                    "bbox": tuple(line.get("bbox", (0, 0, 0, 0))),
                    "avg_font_size": avg_size,
                }
            )

    # Sort by y then x (reasonable reading order for most PDFs)
    lines_out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return lines_out


def _detect_header_footer_zones(
    lines_per_page: List[List[Dict[str, Any]]],
    page_height: float,
    top_pct: float = 0.08,
    bottom_pct: float = 0.08,
    min_repeat_ratio: float = 0.6,
) -> Tuple[set, set]:
    """
    Detect repeating header/footer strings across pages.
    Returns: (header_texts, footer_texts)
    """
    header_counts = {}
    footer_counts = {}
    n_pages = max(1, len(lines_per_page))

    top_y = page_height * top_pct
    bottom_y = page_height * (1.0 - bottom_pct)

    for lines in lines_per_page:
        for ln in lines:
            t = ln["text"].strip()
            y0 = ln["bbox"][1]
            if y0 <= top_y:
                header_counts[t] = header_counts.get(t, 0) + 1
            elif y0 >= bottom_y:
                footer_counts[t] = footer_counts.get(t, 0) + 1

    # Keep texts that repeat across enough pages
    header_texts = {t for t, c in header_counts.items() if c / n_pages >= min_repeat_ratio and len(t) <= 120}
    footer_texts = {t for t, c in footer_counts.items() if c / n_pages >= min_repeat_ratio and len(t) <= 120}

    # Also include common "Page X" patterns
    header_texts |= {t for t in header_counts if _looks_like_header_footer(t)}
    footer_texts |= {t for t in footer_counts if _looks_like_header_footer(t)}

    return header_texts, footer_texts


def _is_heading(line: Dict[str, Any], body_font_size: float, heading_ratio: float = 1.25) -> bool:
    # Heuristic: larger font and not too long
    txt = line["text"].strip()
    if len(txt) > 140:
        return False
    return line["avg_font_size"] >= body_font_size * heading_ratio


def _compute_body_font_size(all_lines: List[Dict[str, Any]]) -> float:
    # Robust-ish: median of font sizes
    sizes = [ln["avg_font_size"] for ln in all_lines if ln["avg_font_size"] > 0]
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _split_into_chunks(
    blocks: List[Dict[str, Any]],
    max_chars: int,
    overlap_chars: int
) -> List[Dict[str, Any]]:
    """
    blocks: list of dicts with 'text', and optional metadata
    Returns list of chunk dicts with combined text and inherited metadata.
    """
    chunks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_meta = None
    cur_len = 0

    def flush():
        nonlocal cur, cur_meta, cur_len
        if not cur:
            return
        text = _normalize("\n".join(cur))
        if text:
            out = dict(cur_meta or {})
            out["text"] = text
            chunks.append(out)
        cur = []
        cur_meta = None
        cur_len = 0

    for b in blocks:
        t = b["text"].strip()
        if not t:
            continue

        # If this is a heading, flush current chunk first and start new section context
        if b.get("is_heading"):
            flush()
            cur_meta = {k: v for k, v in b.items() if k != "text"}
            cur.append(t)
            cur_len = len(t)
            continue

        if cur_meta is None:
            cur_meta = {k: v for k, v in b.items() if k != "text"}

        # Add paragraph, flush if too large
        if cur_len + len(t) + 2 > max_chars:
            # create overlap by keeping the tail of current text
            if overlap_chars > 0 and cur:
                joined = "\n".join(cur)
                tail = joined[-overlap_chars:]
                flush()
                cur_meta = {k: v for k, v in b.items() if k != "text"}
                cur = [tail, t]
                cur_len = len(tail) + len(t)
            else:
                flush()
                cur_meta = {k: v for k, v in b.items() if k != "text"}
                cur = [t]
                cur_len = len(t)
        else:
            cur.append(t)
            cur_len += len(t) + 2

    flush()
    return chunks


# -------------------------
# Optimized extractor
# -------------------------

def extract_chunks_from_pdf(
    pdf_path: str,
    *,
    max_chars: int = 3500,
    overlap_chars: int = 200,
    heading_ratio: float = 1.25,
    remove_repeating_headers_footers: bool = True,
) -> List[Dict[str, Any]]:
    
    #print("chunk_extracto.extrac_chunks_from_pdf")
    #print("pdf_path: ", pdf_path)
    """
    Optimized extraction for RAG:
    - Uses layout-aware line extraction (get_text('dict'))
    - Removes repeating headers/footers across pages
    - Normalizes/dehyphenates
    - Detects headings (font-size heuristic)
    - Chunks along paragraph/heading boundaries (not raw fixed windows)
    - Keeps citations: source_file + page (+ optional section heading)

    Returns: List[Dict[str, Any]] with keys:
      - source_file, page, chunk_index, text
      - section (optional, inferred from headings)
    """
    doc = fitz.open(pdf_path)
    source_file = os.path.basename(pdf_path)

    # Pre-extract lines per page (needed to detect repeating headers/footers)
    lines_per_page: List[List[Dict[str, Any]]] = []
    page_heights: List[float] = []
    all_lines_flat: List[Dict[str, Any]] = []

    for pi in range(doc.page_count):
        page = doc.load_page(pi)
        page_heights.append(float(page.rect.height))
        lines = _iter_lines_with_style(page)
        lines_per_page.append(lines)
        all_lines_flat.extend(lines)

    body_font = _compute_body_font_size(all_lines_flat)
    header_texts, footer_texts = (set(), set())
    if remove_repeating_headers_footers and page_heights:
        # Use median page height for thresholds
        ph = sorted(page_heights)[len(page_heights) // 2]
        header_texts, footer_texts = _detect_header_footer_zones(lines_per_page, ph)

    out_chunks: List[Dict[str, Any]] = []
    running_chunk_index = 0
    current_section: Optional[str] = None

    for page_index, lines in enumerate(lines_per_page):
        # Build "paragraph blocks" from lines using vertical gaps
        blocks: List[Dict[str, Any]] = []

        prev_y1 = None
        para_lines: List[str] = []
        para_bbox: Optional[Tuple[float, float, float, float]] = None

        def flush_para():
            nonlocal para_lines, para_bbox, blocks, current_section
            if not para_lines:
                return
            text = "\n".join(para_lines).strip()
            text = _normalize(text)
            if text:
                blocks.append(
                    {
                        "source_file": source_file,
                        "page": page_index + 1,
                        "section": current_section,
                        "text": text,
                    }
                )
            para_lines = []
            para_bbox = None

        for ln in lines:
            t = ln["text"].strip()
            if not t:
                continue
            if t in header_texts or t in footer_texts:
                continue
            if _looks_like_header_footer(t):
                # keep only if not obviously just page numbering etc.
                continue

            # Heading detection (font size)
            is_head = _is_heading(ln, body_font, heading_ratio=heading_ratio)
            if is_head:
                flush_para()
                current_section = t  # section heading for later chunks
                blocks.append(
                    {
                        "source_file": source_file,
                        "page": page_index + 1,
                        "section": current_section,
                        "is_heading": True,
                        "text": t,
                    }
                )
                prev_y1 = ln["bbox"][3]
                continue

            # Paragraph boundary by vertical gap (simple but effective)
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            if prev_y1 is not None and (y0 - prev_y1) > (body_font * 0.9):
                flush_para()

            para_lines.append(t)
            prev_y1 = y1

        flush_para()

        # Chunk across the page's blocks (keeps section metadata)
        page_chunks = _split_into_chunks(blocks, max_chars=max_chars, overlap_chars=overlap_chars)

        for ch in page_chunks:
            ch["chunk_index"] = running_chunk_index
            running_chunk_index += 1
            out_chunks.append(ch)

    return out_chunks


def create_chunks(scan_dir: str, chunk_dir: str):
    if not os.path.isdir(scan_dir):
        print(f"Scan directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(chunk_dir, exist_ok=True)
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
            chunk_dir,
            os.path.splitext(pdf_file)[0] + ".jsonl"
        )
        print(f"Processing {pdf_file} ...")
        chunks = extract_chunks_from_pdf(pdf_path)
        if not chunks:
            print(f"  ⚠️  No text found in {pdf_file}, skipping")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for row in chunks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  ✔ Wrote {len(chunks)} chunks → {out_path}")
    print("Done.")
