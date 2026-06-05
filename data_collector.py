import requests
from bs4 import BeautifulSoup
import os
import json
import time
import re

# ── Domain filter keywords ──────────────────────────────────────────────────

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

ISSUE_YEAR_MAP = {
    20: 2025, 19: 2024, 18: 2023, 17: 2023, 16: 2022,
    15: 2021, 14: 2020, 13: 2020, 12: 2019, 11: 2019,
    10: 2018, 9: 2017, 8: 2016, 7: 2016, 6: 2016,
    5: 2015, 4: 2015, 3: 2015, 2: 2015, 1: 2015
}

# ── Domain detection ────────────────────────────────────────────────────────

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

# ── Scrape one issue index page ─────────────────────────────────────────────

def scrape_issue(issue_num, division="HCD"):
    div_code = "HD" if division == "HCD" else "AD"
    div_id   = "2"  if division == "HCD" else "1"
    year     = ISSUE_YEAR_MAP.get(issue_num, 2020)

    url = (
        f"https://www.supremecourt.gov.bd/web/"
        f"?page=bulletin/bulletin_list_{div_code}_{issue_num}.php"
        f"&menu=00&div_id={div_id}&issue={issue_num}"
    )

    print(f"Scraping {division} issue {issue_num} ({year})...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch: {e}")
        return []

    soup  = BeautifulSoup(response.content, "lxml")
    table = soup.find("table")

    if not table:
        print("  No table found.")
        return []

    rows  = table.find_all("tr")[1:]
    cases = []

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Serial number
        try:
            serial = int(cells[0].get_text(strip=True).replace(".", ""))
        except:
            continue

        # Citation cell
        citation_cell = cells[1]
        citation_text = citation_cell.get_text(separator=" ", strip=True)

        # PDF URL from known pattern
        filename = f"{serial:02d}.%20{issue_num}%20SCOB%20{division}.pdf"
        pdf_link = (
            f"https://www.supremecourt.gov.bd"
            f"/resources/bulletin/{issue_num}/{filename}"
        )

        # Summary and key ratio
        summary   = cells[2].get_text(separator=" ", strip=True) if len(cells) > 2 else ""
        key_ratio = cells[3].get_text(separator=" ", strip=True) if len(cells) > 3 else ""

        # Detect domain
        full_text = f"{citation_text} {summary}"
        domain    = detect_domain(full_text)

        if domain:
            case = {
                "serial":        serial,
                "issue":         issue_num,
                "year":          year,
                "division":      division,
                "citation":      f"{issue_num} SCOB [{year}] {division} {serial}",
                "domain":        domain,
                "keywords":      citation_text,
                "summary":       summary,
                "key_ratio":     key_ratio,
                "pdf_url":       pdf_link,
                "pdf_file":      f"{domain}_{issue_num}SC{division}{serial:02d}.pdf",
                "outcome":       "unknown",
                "outcome_phrase": None
            }
            cases.append(case)
            print(f"  [{domain.upper()}] {case['citation']}")

    return cases

# ── Download a PDF ──────────────────────────────────────────────────────────

def download_pdf(url, save_path):
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

# ── Main pipeline ───────────────────────────────────────────────────────────

def run():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw",       exist_ok=True)

    # Load existing cases to preserve labels and avoid duplicates
    index_path = "data/processed/case_index.json"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_citations = {c["citation"] for c in existing}
        print(f"Loaded {len(existing)} existing cases\n")
    else:
        existing           = []
        existing_citations = set()

    all_cases = existing.copy()
    new_count = 0

    # Scrape HCD issues 1–20
    print("── Scraping High Court Division ──")
    for issue in range(1, 21):
        cases = scrape_issue(issue, division="HCD")
        for c in cases:
            if c["citation"] not in existing_citations:
                all_cases.append(c)
                existing_citations.add(c["citation"])
                new_count += 1
        time.sleep(2)

    # Scrape Appellate Division issues 1–20
    print("\n── Scraping Appellate Division ──")
    for issue in range(1, 21):
        cases = scrape_issue(issue, division="AD")
        for c in cases:
            if c["citation"] not in existing_citations:
                all_cases.append(c)
                existing_citations.add(c["citation"])
                new_count += 1
        time.sleep(2)

    # Summary
    print(f"\n{'─'*40}")
    print(f"New cases found : {new_count}")
    print(f"Total cases     : {len(all_cases)}")
    for domain in ["land", "contract", "service"]:
        count = sum(1 for c in all_cases if c["domain"] == domain)
        print(f"  {domain.capitalize():10s}: {count} cases")
    print(f"{'─'*40}")

    # Save
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"\nCase index saved → {index_path}")

    # Download PDFs for new cases
    print("\nDownloading PDFs for new cases...")
    ok, fail = 0, 0
    for case in all_cases:
        path = os.path.join("data/raw", case["pdf_file"])
        if os.path.exists(path):
            ok += 1
            continue
        print(f"  Downloading {case['pdf_file']}...")
        if download_pdf(case["pdf_url"], path):
            ok += 1
        else:
            fail += 1
        time.sleep(1)

    print(f"\nDone — Downloaded: {ok}  Failed: {fail}")

if __name__ == "__main__":
    run()