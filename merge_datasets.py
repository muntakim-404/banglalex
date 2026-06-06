import json
import os

# Load existing scraped and labeled cases
with open("data/processed/case_index.json", "r", encoding="utf-8") as f:
    existing = json.load(f)

# Normalize existing cases to match manual format
existing_normalized = [
    {
        "citation":       c["citation"],
        "domain":         c["domain"],
        "year":           c["year"],
        "division":       c["division"],
        "appellant":      c.get("appellant", ""),
        "respondent":     c.get("respondent", ""),
        "facts_summary":  c.get("summary", ""),
        "statutes_cited": c.get("keywords", ""),
        "outcome":        c.get("outcome", "unknown"),
        "notes":          c.get("outcome_phrase", "") or ""
    }
    for c in existing
    if c["domain"] in ["land", "contract", "service"]
]

print(f"Existing cases (land/contract/service): {len(existing_normalized)}")

# Load manually collected cases
with open("manual_cases.json", "r", encoding="utf-8") as f:
    manual = json.load(f)

print(f"Manual cases: {len(manual)}")

# Merge — skip duplicates by citation
existing_citations = {c["citation"] for c in existing_normalized}
new_cases = [c for c in manual if c["citation"] not in existing_citations]
print(f"New unique manual cases: {len(new_cases)}")

# Combine
all_cases = existing_normalized + new_cases

# Summary
print(f"\n{'─'*40}")
print(f"Total cases: {len(all_cases)}")
for domain in ["land", "contract", "service"]:
    count = sum(1 for c in all_cases if c["domain"] == domain)
    print(f"  {domain.capitalize():10s}: {count} cases")
print(f"{'─'*40}")

# Save final dataset
os.makedirs("data/annotated", exist_ok=True)
output_path = "data/annotated/cases.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_cases, f, ensure_ascii=False, indent=2)

print(f"\nSaved → {output_path}")