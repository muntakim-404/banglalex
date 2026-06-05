import json
import re
import os

FAVORABLE_PHRASES = [
    "appeal is allowed", "appeal allowed", "appeals allowed",
    "petition allowed", "petition is allowed",
    "rule is made absolute", "rule made absolute", "rule absolute",
    "application allowed", "revision allowed",
    "suit is decreed", "suit decreed",
    "judgment is set aside", "judgment set aside",
    "order is set aside", "order set aside",
    "decision is set aside", "reversed",
    "direction is issued", "directed to",
    "writ is allowed", "writ allowed",
    "writ petition allowed", "writ petition is allowed",
    "decree is set aside", "decree set aside",
    "conviction set aside", "sentence reduced",
    "matter is remanded", "matter remanded",
    "case is remanded", "case remanded",
    "remand is allowed", "sent back",
    "allowed with costs", "allowed without costs",
    "respondent is directed", "respondents are directed",
    "government is directed", "authority is directed",
    "high court division allowed", "high court allowed",
    "appellate division allowed",
    "injunction granted", "injunction is granted",
    "stay is granted", "stay granted",
    "relief is granted", "relief granted",
    "declaration is granted", "declared",
    "quashed", "quashed and set aside"
]

UNFAVORABLE_PHRASES = [
    "appeal is dismissed", "appeal dismissed", "appeals dismissed",
    "petition dismissed", "petition is dismissed",
    "rule is discharged", "rule discharged",
    "application dismissed", "revision dismissed",
    "suit is dismissed", "suit dismissed",
    "upheld", "affirmed", "no merit",
    "writ is dismissed", "writ dismissed",
    "writ petition dismissed", "writ petition is dismissed",
    "leave refused", "leave is refused",
    "dismissed with costs", "dismissed without costs",
    "high court division dismissed", "high court dismissed",
    "appellate division dismissed",
    "conviction maintained", "sentence maintained",
    "decree confirmed", "order maintained", "order upheld",
    "judgment upheld", "judgment maintained",
    "not entitled", "claim dismissed",
    "no interference", "no merit in the appeal",
    "found no merit", "without merit",
    "rejected", "claim rejected"
]

def detect_outcome(summary, key_ratio):
    text = (summary + " " + key_ratio).lower()

    for phrase in FAVORABLE_PHRASES:
        if phrase in text:
            return "favorable", phrase

    for phrase in UNFAVORABLE_PHRASES:
        if phrase in text:
            return "unfavorable", phrase

    return "unknown", None

def run():
    index_path = "data/processed/case_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = {"favorable": 0, "unfavorable": 0, "unknown": 0}

    for case in cases:
        # Don't overwrite manually set labels
        if case.get("outcome") not in [None, "unknown"]:
            results[case["outcome"]] += 1
            continue

        outcome, matched_phrase = detect_outcome(
            case.get("summary", ""),
            case.get("key_ratio", "")
        )
        case["outcome"]        = outcome
        case["outcome_phrase"] = matched_phrase
        results[outcome] += 1

    for case in cases:
        o      = case["outcome"]
        status = f"[{o.upper()}]"
        phrase = f"← '{case['outcome_phrase']}'" if case.get("outcome_phrase") else "← needs manual review"
        print(f"  {status:15s} {case['citation']} {phrase}")

    print(f"\n{'─'*40}")
    print(f"Favorable  : {results['favorable']}")
    print(f"Unfavorable: {results['unfavorable']}")
    print(f"Unknown    : {results['unknown']} ← these need manual labeling")
    print(f"{'─'*40}")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {index_path}")

if __name__ == "__main__":
    run()