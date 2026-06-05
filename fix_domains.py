import json

DOMAIN_KEYWORDS = {
    "land": [
        "land", "property", "possession", "partition", "ejectment",
        "mutation", "khas", "pre-emption", "trespass", "lease",
        "tenancy", "title deed", "registration", "transfer of property",
        "state acquisition", "vested property", "boundary", "survey"
    ],
    "contract": [
        "contract", "breach", "specific performance", "money decree",
        "agreement", "promissory note", "work order", "tender",
        "specific relief", "arbitration", "supply", "dues", "arrears",
        "unpaid", "non-payment", "commercial", "trade"
    ],
    "service": [
        "service rules", "government servant", "government service",
        "termination of service", "termination", "dismissal",
        "reinstatement", "promotion", "seniority",
        "departmental proceeding", "compulsory retirement",
        "back pay", "arrears of salary", "administrative tribunal",
        "bangladesh service rules", "government employee",
        "civil service", "public service", "service matter",
        "service benefit", "pension", "gratuity", "increment",
        "posting", "transfer order", "suspension"
    ]
}

EXCLUDE_KEYWORDS = [
    "murder", "rape", "robbery", "theft", "penal code",
    "criminal", "jail appeal", "death reference", "narcotic",
    "crpc", "acid", "arson", "session"
]

def detect_domain(text):
    text_lower = text.lower()
    for kw in EXCLUDE_KEYWORDS:
        if kw in text_lower:
            return None
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return domain
    return None

def run():
    index_path = "data/processed/case_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases\n")

    changed = 0
    for case in cases:
        full_text = f"{case.get('keywords', '')} {case.get('summary', '')}"
        new_domain = detect_domain(full_text)

        if new_domain and new_domain != case["domain"]:
            print(f"  {case['citation']}: {case['domain']} → {new_domain}")
            case["domain"] = new_domain
            changed += 1

    print(f"\nDomain updates: {changed}")

    # Remove family cases
    before = len(cases)
    cases = [c for c in cases if c["domain"] != "family"]
    removed = before - len(cases)
    if removed:
        print(f"Removed {removed} family cases")

    # Summary
    print(f"\n{'─'*40}")
    print(f"Total cases: {len(cases)}")
    for domain in ["land", "contract", "service"]:
        count = sum(1 for c in cases if c["domain"] == domain)
        print(f"  {domain.capitalize():10s}: {count} cases")
    print(f"{'─'*40}")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {index_path}")

if __name__ == "__main__":
    run()