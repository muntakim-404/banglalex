"""
generate_form_content.py — BanglaLex annotation verification
Draws a stratified random sample (domain x outcome) of 25 real cases,
writes blinded case text for the Google Form and a private answer key.

Usage:
    python generate_form_content.py
"""

import json
import random
import csv

INPUT_PATH  = "data/annotated/cases.json"   # the 164 real cases
SAMPLE_SIZE = 25
SEED        = 42
FORM_OUT    = "form_content.txt"
ANSWER_OUT  = "answer_key.csv"

random.seed(SEED)


def stratified_sample(cases, n):
    strata = {}
    for c in cases:
        key = (c.get("domain"), c.get("outcome"))
        strata.setdefault(key, []).append(c)

    total = len(cases)
    raw_alloc = {k: len(v) / total * n for k, v in strata.items()}
    alloc = {k: int(v) for k, v in raw_alloc.items()}
    remainder = n - sum(alloc.values())

    fracs = sorted(raw_alloc.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for k, _ in fracs[:remainder]:
        alloc[k] += 1

    sample = []
    for k, count in alloc.items():
        pool = strata[k]
        count = min(count, len(pool))
        sample.extend(random.sample(pool, count))

    if len(sample) < n:
        remaining = [c for c in cases if c not in sample]
        sample.extend(random.sample(remaining, n - len(sample)))

    random.shuffle(sample)
    return sample[:n]


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    real_cases = [c for c in cases if not c.get("is_augmented")]
    print(f"Real cases available: {len(real_cases)}")

    def is_clean(c):
        if not c.get("appellant", "").strip():
            return False
        if not c.get("respondent", "").strip():
            return False
        statutes = c.get("statutes_cited", "")
        if "Vs." in statutes or "SCOB" in statutes:
            return False
        if len(statutes) > 300:   # contaminated fields run long
            return False
        return True

    clean_cases = [c for c in real_cases if is_clean(c)]
    dropped = len(real_cases) - len(clean_cases)
    print(f"Dropped {dropped} cases with malformed fields "
          f"(blank appellant/respondent or contaminated statutes_cited)")
    print(f"Clean cases available: {len(clean_cases)}")

    sample = stratified_sample(clean_cases, SAMPLE_SIZE)
    print(f"Sampled: {len(sample)}")

    by_domain = {}
    for c in sample:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
    print("Domain breakdown:", by_domain)

    by_outcome = {}
    for c in sample:
        by_outcome[c["outcome"]] = by_outcome.get(c["outcome"], 0) + 1
    print("Outcome breakdown:", by_outcome)

    # Blinded form content — no citation, no year, no outcome
    with open(FORM_OUT, "w", encoding="utf-8") as f:
        for i, c in enumerate(sample, 1):
            f.write(f"Case {i}\n")
            f.write(f"Petitioner/Appellant: {c.get('appellant', 'Unknown')}\n")
            f.write(f"Respondent: {c.get('respondent', 'Unknown')}\n\n")
            f.write(f"Facts: {c.get('facts_summary', '')}\n\n")
            f.write(f"Statutes cited: {c.get('statutes_cited', '')}\n")
            f.write("\n" + "=" * 60 + "\n\n")

    # Private answer key — do not share with students
    with open(ANSWER_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_number", "citation", "domain", "original_outcome"])
        for i, c in enumerate(sample, 1):
            writer.writerow([i, c["citation"], c["domain"], c["outcome"]])

    print(f"\nSaved -> {FORM_OUT}  (paste into Google Form)")
    print(f"Saved -> {ANSWER_OUT}  (keep private)")


if __name__ == "__main__":
    main()
