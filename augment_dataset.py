"""
augment_dataset.py — BanglaLex Phase 2: Data Augmentation
Target : 400 total cases (164 real + 236 augmented)
Domains: Land→133  Contract→133  Service→134

Uses Groq API (free tier) for LLM-based paraphrasing.
Get a free key at: https://console.groq.com

Setup:
    pip install groq
    $env:GROQ_API_KEY = "gsk_..."   # PowerShell
    python augment_dataset.py
"""

import json
import os
import time
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    print("Run: pip install groq")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
MODEL           = "llama-3.1-8b-instant"   # fast + free
                                            # alternative: "llama-3.3-70b-versatile"
DOMAIN_TARGETS  = {"land": 133, "contract": 133, "service": 134}
INPUT_PATH      = "data/annotated/cases.json"
OUTPUT_PATH     = "data/annotated/cases_augmented.json"
CHECKPOINT_PATH = "data/annotated/aug_checkpoint.json"
DELAY           = 0.8   # seconds between calls (stay under free-tier rate limit)
# ─────────────────────────────────────────────────────────────────────────────


def augment_summary(client, summary: str, domain: str, copy_num: int) -> str:
    """Paraphrase a case facts summary using Groq."""
    prompt = (
        f"Rewrite the following {domain} law case summary using completely different "
        "words and sentence structures. Preserve all legal facts, parties, statutes, "
        "and the outcome exactly. Do not add or remove any information. "
        "Output only the rewritten summary — no preamble, no explanation.\n\n"
        f"Original:\n{summary}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.75 + (copy_num - 1) * 0.05,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"\n  [retry {attempt+1}/3 after {wait}s] {e}")
            time.sleep(wait)
    raise RuntimeError("Failed after 3 retries")


def make_aug_case(original: dict, new_summary: str, copy_num: int) -> dict:
    aug = original.copy()
    aug["citation"]        = f"AUG{copy_num}-{original['citation']}"
    aug["facts_summary"]   = new_summary
    aug["is_augmented"]    = True
    aug["source_citation"] = original["citation"]
    aug["notes"]           = (
        f"[Augmented copy {copy_num} of {original['citation']}] "
        + original.get("notes", "")
    )
    return aug


def load_checkpoint() -> list:
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint found: {len(data)} augmented cases already done.")
        return data
    return []


def save_checkpoint(augmented: list):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)


def main():
    # ── Preflight ──────────────────────────────────────────────────────────
    if not GROQ_API_KEY:
        print("❌  GROQ_API_KEY not set.")
        print("    PowerShell: $env:GROQ_API_KEY = 'gsk_...'")
        print("    Free key  : https://console.groq.com")
        return

    client = Groq(api_key=GROQ_API_KEY)

    # ── Load dataset ───────────────────────────────────────────────────────
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        real_cases = json.load(f)

    print(f"Real cases: {len(real_cases)}")
    by_domain = {
        d: [c for c in real_cases if c.get("domain") == d]
        for d in DOMAIN_TARGETS
    }
    for d, cases in by_domain.items():
        print(f"  {d}: {len(cases)}")

    total_needed = sum(
        t - len(by_domain[d]) for d, t in DOMAIN_TARGETS.items()
    )
    print(f"\nAugmented cases needed: {total_needed}")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    augmented      = load_checkpoint()
    done_citations = {a["citation"] for a in augmented}

    # ── Augment domain by domain ───────────────────────────────────────────
    for domain, target in DOMAIN_TARGETS.items():
        cases      = by_domain[domain]
        real_count = len(cases)
        aug_needed = target - real_count

        full_passes = aug_needed // real_count
        remainder   = aug_needed  % real_count

        print(
            f"\n── {domain.upper()} ──  "
            f"real={real_count}  need={aug_needed}  "
            f"({full_passes} full pass{'es' if full_passes != 1 else ''}"
            f"{f' + {remainder} extra' if remainder else ''})"
        )

        copy_num = 1

        # Full passes — every case gets one copy per pass
        for _pass in range(full_passes):
            for i, case in enumerate(cases):
                key = f"AUG{copy_num}-{case['citation']}"
                if key in done_citations:
                    continue
                print(
                    f"  [copy {copy_num}] [{i+1:>3}/{real_count}] "
                    f"{case['citation'][:45]}",
                    end="\r"
                )
                try:
                    new_summary = augment_summary(
                        client, case["facts_summary"], domain, copy_num
                    )
                    aug = make_aug_case(case, new_summary, copy_num)
                    augmented.append(aug)
                    done_citations.add(key)
                    save_checkpoint(augmented)
                    time.sleep(DELAY)
                except Exception as e:
                    print(f"\n  ❌  {case['citation']}: {e}")
            copy_num += 1

        # Partial pass — pick a balanced subset for the remainder
        if remainder > 0:
            fav = [c for c in cases if c.get("outcome") == "favorable"]
            unf = [c for c in cases if c.get("outcome") == "unfavorable"]
            subset, fi, ui = [], 0, 0
            while len(subset) < remainder:
                if fi < len(fav):
                    subset.append(fav[fi]); fi += 1
                if len(subset) < remainder and ui < len(unf):
                    subset.append(unf[ui]); ui += 1

            for i, case in enumerate(subset):
                key = f"AUG{copy_num}-{case['citation']}"
                if key in done_citations:
                    continue
                print(
                    f"  [copy {copy_num}/partial] [{i+1:>3}/{remainder}] "
                    f"{case['citation'][:40]}",
                    end="\r"
                )
                try:
                    new_summary = augment_summary(
                        client, case["facts_summary"], domain, copy_num
                    )
                    aug = make_aug_case(case, new_summary, copy_num)
                    augmented.append(aug)
                    done_citations.add(key)
                    save_checkpoint(augmented)
                    time.sleep(DELAY)
                except Exception as e:
                    print(f"\n  ❌  {case['citation']}: {e}")

        domain_done = len([a for a in augmented if a.get("domain") == domain])
        print(f"\n  ✓ {domain}: {domain_done} augmented")

    # ── Save final dataset ─────────────────────────────────────────────────
    all_cases = real_cases + augmented

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint
    if Path(CHECKPOINT_PATH).exists():
        os.remove(CHECKPOINT_PATH)
        print("\nCheckpoint removed.")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"Total cases: {len(all_cases)}")
    print(f"{'Domain':<12} {'Real':>6} {'Augmented':>10} {'Total':>7}")
    print(f"{'─'*37}")
    for domain in ["land", "contract", "service"]:
        r = len([c for c in real_cases if c.get("domain") == domain])
        a = len([c for c in augmented  if c.get("domain") == domain])
        print(f"{domain.capitalize():<12} {r:>6} {a:>10} {r+a:>7}")
    print(f"{'─'*37}")
    r_total = len(real_cases)
    a_total = len(augmented)
    print(f"{'Total':<12} {r_total:>6} {a_total:>10} {r_total+a_total:>7}")

    fav = sum(1 for c in all_cases if c.get("outcome") == "favorable")
    unf = sum(1 for c in all_cases if c.get("outcome") == "unfavorable")
    print(f"\nFavorable: {fav}  Unfavorable: {unf}  "
          f"Ratio: {fav/(fav+unf):.2f}")
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()