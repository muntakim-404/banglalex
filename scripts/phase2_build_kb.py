"""
Phase 2 — Build BanglaLex Knowledge Base
==========================================
Runs the full pipeline in four steps:
  1. Load  → read processed_law.json from --data-dir
  2. Chunk → section-level sliding-window split  (saved to chunks.json)
  3. Embed → paraphrase-multilingual-MiniLM-L12-v2 (CUDA auto-detected)
  4. Index → FAISS IndexFlatIP, saved to --kb-dir

Usage (defaults work for your project layout):
    python scripts/phase2_build_kb.py --data-dir data/statutes

Optional flags:
    --kb-dir     data/knowledge_base   output directory for FAISS index
    --max-words  200                   words per chunk
    --overlap    30                    sliding-window overlap
    --batch-size 64                    reduce to 32 if OOM on RTX 3050
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.knowledge_base.chunker  import load_processed_law, chunk_acts, save_chunks
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.indexer  import KnowledgeBaseIndex

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)
SEP = "─" * 60


def parse_args():
    p = argparse.ArgumentParser(
        description="Build BanglaLex knowledge base (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   default="data/statutes",
                   help="Directory containing processed_law.json")
    p.add_argument("--kb-dir",     default="data/knowledge_base",
                   help="Output directory for FAISS index + metadata")
    p.add_argument("--chunks-path", default=None,
                   help="Where to save chunks.json (default: kb-dir/chunks.json)")
    p.add_argument("--max-words",  type=int, default=200)
    p.add_argument("--overlap",    type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Embedding batch size — reduce to 32 if OOM")
    p.add_argument("--model",
                   default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--keep-repealed", action="store_true",
                   help="Include repealed acts in the knowledge base")
    return p.parse_args()


def main():
    args        = parse_args()
    data_dir    = Path(args.data_dir)
    kb_dir      = Path(args.kb_dir)
    chunks_path = Path(args.chunks_path) if args.chunks_path else kb_dir / "chunks.json"
    wall_start  = time.time()

    logger.info(SEP)
    logger.info("BanglaLex — Phase 2: Knowledge Base Build")
    logger.info(SEP)
    logger.info(f"  Data dir    : {data_dir}")
    logger.info(f"  KB output   : {kb_dir}")
    logger.info(f"  Max words   : {args.max_words}  |  Overlap : {args.overlap}")
    logger.info(f"  Batch size  : {args.batch_size}  |  Model  : {args.model}")
    logger.info(f"  Skip repealed: {not args.keep_repealed}")
    logger.info(SEP)

    # ── Step 1: Load ──────────────────────────────────────────────
    logger.info("Step 1/4 — Loading processed_law.json")
    acts = load_processed_law(data_dir)

    # ── Step 2: Chunk ─────────────────────────────────────────────
    logger.info(SEP)
    logger.info("Step 2/4 — Chunking sections")
    t0     = time.time()
    chunks = chunk_acts(
        acts,
        max_words     = args.max_words,
        overlap       = args.overlap,
        skip_repealed = not args.keep_repealed,
    )
    logger.info(f"  {len(chunks):,} chunks in {time.time()-t0:.1f}s")

    # Top acts by chunk count
    act_counts = Counter(c["act_name"] for c in chunks)
    logger.info("  Top 8 acts by chunk count:")
    for act, cnt in act_counts.most_common(8):
        logger.info(f"    {cnt:>5}  {act[:70]}")

    save_chunks(chunks, chunks_path)

    # ── Step 3: Embed ─────────────────────────────────────────────
    logger.info(SEP)
    logger.info("Step 3/4 — Generating embeddings  (this is the slow part)")
    embedder = Embedder(args.model)
    texts    = [c["text"] for c in chunks]

    t0         = time.time()
    embeddings = embedder.embed(texts, batch_size=args.batch_size, show_progress=True)
    elapsed    = time.time() - t0
    logger.info(f"  Embedded {len(texts):,} chunks in {elapsed:.1f}s  "
                f"({len(texts)/elapsed:.0f} chunks/s)")
    logger.info(f"  Embedding matrix: {embeddings.shape}  dtype={embeddings.dtype}")

    # ── Step 4: Index ─────────────────────────────────────────────
    logger.info(SEP)
    logger.info("Step 4/4 — Building FAISS index")
    t0 = time.time()
    kb = KnowledgeBaseIndex(dim=embedder.dim)
    kb.build(embeddings, chunks)
    kb.save(kb_dir)
    logger.info(f"  Index built in {time.time()-t0:.1f}s")
    logger.info(f"  Stats: {kb.stats()}")

    # ── Sanity check ──────────────────────────────────────────────
    logger.info(SEP)
    logger.info("Sanity check — two test queries")
    from src.knowledge_base.retriever import Retriever
    retriever = Retriever(kb, embedder)
    for q in [
        "punishment for theft under Bangladesh Penal Code",
        "ভূমি বিরোধ সীমানা নির্ধারণ আইন",
    ]:
        results = retriever.retrieve(q, k=2)
        logger.info(f"\n  Query: '{q}'")
        for i, r in enumerate(results, 1):
            logger.info(
                f"    [{i}] {r['act_name'][:60]}  §{r['section_no']} "
                f"score={r['score']:.4f}"
            )

    # ── Done ──────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info(f"Phase 2 complete in {time.time()-wall_start:.1f}s")
    logger.info(f"  {kb_dir / 'faiss_index.bin'}")
    logger.info(f"  {kb_dir / 'metadata.json'}")
    logger.info(f"  {chunks_path}")
    logger.info("")
    logger.info("Next:  python scripts/phase2_test_retrieval.py")
    logger.info(SEP)


if __name__ == "__main__":
    main()
