'''
Created on 29 Jan 2026

@author: ante
'''
#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF


RE_MODULE = re.compile(r"^(module|modul)\s+\d+\b", re.IGNORECASE)
RE_STEP = re.compile(r"^step\s+\d+\b", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^(chapter|kapitel)\s+\d+\b", re.IGNORECASE)
RE_NUMBERED = re.compile(r"^\d+(\.\d+){0,8}\b")  # 1, 1.2, 1.2.3...


def normalize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_probably_bold(fontname: str) -> bool:
    return "bold" in (fontname or "").lower() or "demi" in (fontname or "").lower()


@dataclass
class Heading:
    title: str
    level: int
    page: int
    score: float


def estimate_body_font_size(doc: fitz.Document) -> float:
    """
    Estimate the dominant body font size by sampling spans with reasonably long text.
    """
    sizes: List[float] = []
    for page in doc:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = (span.get("text") or "").strip()
                    if len(t) < 25:
                        continue
                    sz = span.get("size")
                    if sz:
                        sizes.append(round(float(sz), 1))

    if not sizes:
        return 10.0

    c = Counter(sizes)
    return float(c.most_common(1)[0][0])


def line_features_from_spans(spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    text = normalize("".join(s.get("text", "") for s in spans))
    if not text:
        return None

    sizes = [float(s["size"]) for s in spans if s.get("size")]
    fonts = [s.get("font", "") for s in spans]
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0
    bold = any(is_probably_bold(f) for f in fonts)

    is_all_caps = text.isupper() and any(c.isalpha() for c in text)
    ends_with_period = text.endswith(".")
    word_count = len(text.split())
    looks_short = word_count <= 14

    numbered = bool(RE_NUMBERED.match(text) or RE_STEP.match(text) or RE_MODULE.match(text) or RE_CHAPTER.match(text))

    return {
        "text": text,
        "avg_size": avg_size,
        "bold": bold,
        "all_caps": is_all_caps,
        "ends_with_period": ends_with_period,
        "looks_short": looks_short,
        "numbered": numbered,
    }


def heading_score(feat: Dict[str, Any], body_size: float) -> float:
    """
    Heuristic score: larger-than-body + bold + numbering + title-ish shortness.
    """
    score = 0.0
    score += max(0.0, feat["avg_size"] - body_size) * 1.35
    if feat["bold"]:
        score += 1.5
    if feat["numbered"]:
        score += 1.2
    if feat["looks_short"]:
        score += 0.6
    if feat["all_caps"]:
        score += 0.6
    if feat["ends_with_period"]:
        score -= 1.0
    return score


def assign_level(feat: Dict[str, Any], body_size: float) -> int:
    """
    Decide H level. Patterns first, then size deltas.
    """
    txt = feat["text"]

    # explicit patterns first
    if RE_MODULE.match(txt):
        return 1
    if RE_CHAPTER.match(txt):
        return 1
    if RE_STEP.match(txt):
        return 2
    if RE_NUMBERED.match(txt):
        # deeper levels by dot count: "1"->2, "1.2"->3, "1.2.3"->4...
        token = txt.split()[0]
        dots = token.count(".")
        return min(4, 2 + dots)

    delta = feat["avg_size"] - body_size
    if delta >= 4:
        return 1
    if delta >= 2:
        return 2
    if delta >= 1 and (feat["bold"] or feat["numbered"]):
        return 3
    return 0


def extract_headings(pdf_path: Path, min_score: float) -> List[Heading]:
    doc = fitz.open(str(pdf_path))
    body = estimate_body_font_size(doc)

    headings: List[Heading] = []
    for page_idx, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                feat = line_features_from_spans(spans)
                if not feat:
                    continue
                score = heading_score(feat, body)
                if score < min_score:
                    continue
                level = assign_level(feat, body)
                if level == 0:
                    continue

                headings.append(Heading(
                    title=feat["text"],
                    level=level,
                    page=page_idx,
                    score=round(score, 2),
                ))

    # Deduplicate immediate repeats (very common with headers/footers)
    deduped: List[Heading] = []
    prev_key = None
    for h in headings:
        key = (h.title.lower(), h.level, h.page)
        if key == prev_key:
            continue
        deduped.append(h)
        prev_key = key

    return deduped


def build_tree(headings: List[Heading]) -> Dict[str, Any]:
    root: Dict[str, Any] = {"title": "ROOT", "level": 0, "page": None, "score": None, "children": []}
    stack: List[Dict[str, Any]] = [root]

    for h in headings:
        node = {"title": h.title, "level": h.level, "page": h.page, "score": h.score, "children": []}
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        stack[-1]["children"].append(node)
        stack.append(node)

    return root


def tree_to_markdown(node: Dict[str, Any]) -> str:
    lines: List[str] = []

    def walk(n: Dict[str, Any]):
        for ch in n.get("children", []):
            lvl = int(ch["level"])
            prefix = "#" * max(1, min(6, lvl))
            page = ch.get("page")
            title = ch.get("title", "")
            lines.append(f"{prefix} {title} (p. {page})")
            walk(ch)

    walk(node)
    return "\n".join(lines).strip() + "\n"


def slim_tree(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove ROOT and keep a compact JSON suitable for feeding to an LLM.
    """
    return {"items": node.get("children", [])}


def load_all_trees(outline_json_paths: List[Path]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {"pdfs": []}
    for p in outline_json_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        combined["pdfs"].append(data)
    return combined


def  gen_outline(in_dir, out_dir) -> int: # IGNORE:C0111
    """
    ap = argparse.ArgumentParser(
        description="Extract heading hierarchy from all PDFs in a directory (PyMuPDF)."
    )
    ap.add_argument("--in", dest="in_dir", required=True, help="Input directory containing PDFs")
    ap.add_argument("--out", dest="out_dir", default="out", help="Output directory (default: out)")
    ap.add_argument("--min-score", dest="min_score", type=float, default=2.2,
                    help="Heading score threshold (default: 2.2). Increase if too many false headings.")
    ap.add_argument("--glob", dest="glob_pat", default="*.pdf", help="Glob pattern (default: *.pdf)")
    args = ap.parse_args()
    """
    
    min_score = 2.2
    
    in_dir = Path(in_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in %s matching *.pdf ", in_dir)

    all_outline_jsons: List[Path] = []
    combined_md_lines: List[str] = []

    for pdf in pdfs:
        headings = extract_headings(pdf, min_score=min_score)
        tree = build_tree(headings)
        md = tree_to_markdown(tree)

        base = pdf.stem
        json_path = out_dir / f"{base}.outline.json"
        md_path = out_dir / f"{base}.outline.md"

        payload = {
            "pdf": pdf.name,
            "path": str(pdf),
            "min_score": min_score,
            "headings_count": len(headings),
            "outline": slim_tree(tree),
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(md, encoding="utf-8")

        all_outline_jsons.append(json_path)

        combined_md_lines.append(f"# {pdf.name}")
        combined_md_lines.append(md)
        combined_md_lines.append("")  # spacer

        print(f"[OK] {pdf.name}: {len(headings)} headings -> {md_path.name}, {json_path.name}")

    # Combined outputs
    combined_md = "\n".join(combined_md_lines).strip() + "\n"
    (out_dir / "ALL.outlines.md").write_text(combined_md, encoding="utf-8")

    combined_json = load_all_trees(all_outline_jsons)
    (out_dir / "ALL.outlines.json").write_text(json.dumps(combined_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] Wrote combined: ALL.outlines.md, ALL.outlines.json to {out_dir}")


#if __name__ == "__main__":
#    main()