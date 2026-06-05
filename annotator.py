import json
import os

def run():
    index_path = "data/processed/case_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    unknowns = [c for c in cases if c.get("outcome") == "unknown"]
    total    = len(unknowns)

    if total == 0:
        print("No unknown cases — all labeled!")
        return

    print(f"{total} cases need manual labeling.")
    print("Commands:  f = favorable   u = unfavorable   s = skip\n")

    for i, case in enumerate(unknowns):
        print(f"{'═'*60}")
        print(f"[{i+1}/{total}] {case['citation']} | Domain: {case['domain'].upper()}")
        print(f"{'─'*60}")
        print("SUMMARY:")
        print(case.get("summary", "N/A"))
        print(f"\nKEY RATIO:")
        print(case.get("key_ratio", "N/A"))
        print(f"{'─'*60}")

        while True:
            choice = input("Label (f/u/s): ").strip().lower()
            if choice == "f":
                case["outcome"]        = "favorable"
                case["outcome_phrase"] = "manual"
                print("  → Labeled: FAVORABLE\n")
                break
            elif choice == "u":
                case["outcome"]        = "unfavorable"
                case["outcome_phrase"] = "manual"
                print("  → Labeled: UNFAVORABLE\n")
                break
            elif choice == "s":
                print("  → Skipped\n")
                break
            else:
                print("  Invalid. Enter f, u, or s.")

        # Save after every annotation
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)

    # Final summary
    labeled   = sum(1 for c in cases if c.get("outcome") in ["favorable", "unfavorable"])
    remaining = sum(1 for c in cases if c.get("outcome") == "unknown")

    print(f"{'═'*60}")
    print(f"Labeled  : {labeled}")
    print(f"Remaining: {remaining} unknowns")
    print(f"Saved → {index_path}")

if __name__ == "__main__":
    run()