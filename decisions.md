# BanglaLex — Project Decisions

## Decision 1 — Civil Domains
**Chosen domains:** Land disputes, Contract disputes, Family disputes
**Why:** High-frequency civil categories in the Supreme Court 
bulletin, ensuring sufficient case volume for dataset construction 
(~65-70 cases per domain, 200 total minimum).

## Decision 2 — Outcome Labels
**Label space:** Binary — Favorable / Unfavorable
**Why:** Binary classification avoids ambiguous annotation in 
"partially allowed" cases and produces cleaner evaluation metrics.

## Decision 3 — LLM Provider
**Status:** Deferred to Phase 3
**Options being considered:** Groq, Google Gemini API, OpenAI GPT-4o-mini
**Will decide based on:** Bangla language quality testing in Phase 3

## Decision 4 — Dataset Target
**Minimum cases:** 200 (approx. 65-70 per domain)
**Source:** Supreme Court of Bangladesh bulletin
**Exit criterion:** Phase 1 is not complete until 200 labeled 
cases exist in data/annotated/

## Decision 5 — Embedding Model
**Model:** paraphrase-multilingual-MiniLM-L12-v2
**Why:** Lightweight, multilingual, fits within 4GB VRAM, 
handles Bangla text well.