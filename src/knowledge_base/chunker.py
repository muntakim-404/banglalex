"""
BanglaLex — Chunker  (v2: JSON format)
========================================
Loads processed_law.json from the Bangladesh Legal Acts Kaggle dataset
(sakhadib/Bangladesh-Legal-Acts-Dataset) and produces section-level chunks.

Dataset structure (confirmed from inspection):
  processed_law.json
    └── dict
          ├── metadata  →  {total_acts: 1484, total_sections: 35633, ...}
          └── acts      →  list of act dicts, each with:
                act_title, act_no, act_year, publication_date,
                sections  →  list of {section_content: "..."},
                csv_metadata  →  {is_repealed: bool, act_title_from_csv, ...}

Chunking strategy
-----------------
• Each section becomes 1+ chunks (sliding window if > max_words).
• Repealed acts are skipped by default (avoids citing dead law).
• Non-breaking spaces (\xa0) and zero-width spaces are cleaned.
• Max 200 words ≈ 128 word-pieces for paraphrase-multilingual-MiniLM-L12-v2.
"""

import re
import json
import logging
from pathlib import Path
from typing  import Dict, List

logger = logging.getLogger(__name__)

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a legal text string:
      - Replace non-breaking space (\xa0) with regular space
      - Remove zero-width space (\u200b)
      - Collapse all whitespace runs to a single space
      - Strip leading/trailing whitespace
    Returns "" for non-string or NaN input.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def sliding_window(text: str, max_words: int = 200, overlap: int = 30) -> List[str]:
    """
    Split *text* into overlapping word-windows of *max_words* each.
    Returns [text] unchanged if len(text) ≤ max_words words.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks, step = [], max_words - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_processed_law(data_dir: Path) -> List[Dict]:
    """
    Load processed_law.json and return the raw list of act dicts.

    Parameters
    ----------
    data_dir : Path
        The directory containing processed_law.json
        (i.e. data/statutes/)

    Returns
    -------
    List[Dict]  — one dict per legal act
    """
    data_dir  = Path(data_dir)
    law_file  = data_dir / "processed_law.json"

    if not law_file.exists():
        raise FileNotFoundError(
            f"processed_law.json not found in {data_dir}\n"
            "Make sure you extracted the Kaggle dataset into that folder."
        )

    logger.info(f"Loading {law_file}  ({law_file.stat().st_size // 1024 / 1024:.1f} MB) …")
    with open(law_file, encoding="utf-8") as f:
        data = json.load(f)

    acts = data.get("acts", [])
    meta = data.get("metadata", {})
    logger.info(
        f"Loaded {len(acts)} acts | "
        f"{meta.get('total_sections', '?')} sections | "
        f"{meta.get('total_footnotes', '?')} footnotes"
    )
    return acts


# Alias kept for any code that still calls load_kaggle_dataset
load_kaggle_dataset = load_processed_law


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_acts(
    acts:           List[Dict],
    max_words:      int  = 200,
    overlap:        int  = 30,
    skip_repealed:  bool = True,
) -> List[Dict]:
    """
    Convert a list of act dicts into a flat list of chunk dicts.

    Each chunk contains:
        chunk_id, text, act_name, act_no, act_year,
        section_no, section_title, word_count, window_idx

    Parameters
    ----------
    acts          : from load_processed_law()
    max_words     : max words per chunk (200 ≈ 128 MiniLM tokens)
    overlap       : sliding-window word overlap between consecutive chunks
    skip_repealed : if True, acts marked is_repealed=True are excluded
    """
    chunks:            List[Dict] = []
    chunk_id:          int        = 0
    skipped_repealed:  int        = 0
    skipped_empty:     int        = 0

    for act in acts:
        # ── Repealed check ────────────────────────────────────────
        csv_meta    = act.get("csv_metadata", {}) or {}
        is_repealed = csv_meta.get("is_repealed", False)
        if skip_repealed and is_repealed:
            skipped_repealed += 1
            continue

        # ── Act-level metadata ────────────────────────────────────
        act_title = clean_text(act.get("act_title", ""))
        # act_no / act_year may be empty in the JSON — fall back to csv_metadata
        act_no    = clean_text(
            str(act.get("act_no", "") or csv_meta.get("act_no_from_csv",  ""))
        )
        act_year  = clean_text(
            str(act.get("act_year","") or csv_meta.get("act_year_from_csv",""))
        )

        sections = act.get("sections", []) or []
        if not sections:
            skipped_empty += 1
            continue

        # ── Section-level chunking ────────────────────────────────
        for sec_idx, section in enumerate(sections):
            if not isinstance(section, dict):
                continue

            content = clean_text(section.get("section_content", ""))
            if not content:
                continue

            # Section number / title (may or may not exist as separate keys)
            sec_no    = clean_text(
                str(section.get("section_no", section.get("section_number", "")))
            ) or str(sec_idx + 1)            # fallback: use index
            sec_title = clean_text(
                section.get("section_title", section.get("title", ""))
            )

            for win_idx, win_text in enumerate(
                sliding_window(content, max_words=max_words, overlap=overlap)
            ):
                chunks.append({
                    "chunk_id":      chunk_id,
                    "text":          win_text,
                    "act_name":      act_title,
                    "act_no":        act_no,
                    "act_year":      act_year,
                    "section_no":    sec_no,
                    "section_title": sec_title,
                    "word_count":    word_count(win_text),
                    "window_idx":    win_idx,
                })
                chunk_id += 1

    logger.info(
        f"Chunking done — {chunk_id:,} chunks produced | "
        f"{skipped_repealed} repealed acts skipped | "
        f"{skipped_empty} acts with no sections skipped"
    )
    return chunks


# Alias for backward compatibility with the old build script
def chunk_dataframe(acts, max_words=200, overlap=30):
    """Alias → chunk_acts (acts is a list here, not a DataFrame)."""
    return chunk_acts(acts, max_words=max_words, overlap=overlap)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_chunks(chunks: List[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks saved → {path}  ({len(chunks):,} chunks)")


def load_chunks(path: Path) -> List[Dict]:
    with open(Path(path), encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Chunks loaded ← {path}  ({len(chunks):,} chunks)")
    return chunks
