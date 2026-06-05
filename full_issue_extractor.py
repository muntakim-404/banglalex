import pdfplumber
import requests
import os
import json
import re
import time

ISSUE_YEAR_MAP = {
    20: 2025, 19: 2024, 18: 2023, 17: 2023, 16: 2022,
    15: 2021, 14: 2020, 13: 2020, 12: 2019, 11: 2019,
    10: 2018, 9: 2017, 8: 2016, 7: 2016, 6: 2016
}

BASE_URL = "https://www.supremecourt.gov.bd/resources/bulletin"

# ── Download full issue PDF ─────────────────────────────────────────────────

def download_full_issue(issue_num):
    year  = ISSUE_YEAR_MAP.get(issue_num)
    url   = f"{BASE_URL}/{issue_num}_SCOB_{year}.pdf"
    path  = f"data/raw/full_issue_{issue_num}.pdf"

    if os.path.exists(path):
        print(f"  Already exists: full_issue_{issue_num}.pdf")
        return path

    print(f"  Downloading full issue {issue_num} ({year})...")
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Saved ({len(r.content)//1024} KB)")
        return path
    except Exception as e:
        print(f"  Failed: {e}")
        return None

# ── Extract case texts from full PDF ───────────────────────────────────────

def extract_cases_from_pdf(pdf_path, target_cases):
    print(f"  Reading {pdf_path}...")

    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
                if (i + 1) % 50 == 0:
                    print(f"    Read {i+1} pages...")
    except Exception as e:
        print(f"  Could not read PDF: {e}")
        return {}

    print(f"  Total pages: {len(pages_text)}")

    # Find where each case starts using citation pattern
    # e.g. "16 SCOB [2022] HCD 4"
    citation_re = re.compile(
        r'(\d+)\s+SCOB\s+\[(\d+)\]\s+HCD\s+(\d+)',
        re.IGNORECASE
    )

    # Map (issue, serial) → starting page index
    case_start_pages = {}
    for page_idx, text in enumerate(pages_text):
        for m in citation_re.finditer(text):
            issue_found  = int(m.group(1))
            serial_found = int(m.group(3))
            key = (issue_found, serial_found)
            if key not in case_start_pages:
                case_start_pages[key] = page_idx

    # Sort by page to determine end of each case
    sorted_cases = sorted(case_start_pages.items(), key=lambda x: x[1])

    # Build page range for each case
    case_page_ranges = {}
    for i, (key, start) in enumerate(sorted_cases):
        end = sorted_cases[i + 1][1] if i + 1 < len(sorted_cases) else len(pages_text)
        case_page_ranges[key] = (start, end)

    # Extract text for target cases only
    extracted = {}
    for case in target_cases:
        key = (case["issue"], case["serial"])
        if key in case_page_ranges:
            start, end = case_page_ranges[key]
            case_text = "\n".join(pages_text[start:end])
            extracted[case["citation"]] = case_text
            print(f"    ✓ {case['citation']} ({end - start} pages)")
        else:
            print(f"    ✗ Not found: {case['citation']}")

    return extracted

# ── Main ────────────────────────────────────────────────────────────────────

def run():
    index_path = "data/processed/case_index.json"
    if not os.path.exists(index_path):
        print("Run data_collector.py first.")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    print(f"Loaded {len(all_cases)} cases from index\n")

    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load existing extracted texts if any
    output_path = "data/processed/extracted_texts.json"
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_extracted = json.load(f)
    else:
        all_extracted = {}

    # Process issues 15–18 (individual PDFs don't exist for these)
    for issue_num in range(15, 19):
        issue_cases = [c for c in all_cases if c["issue"] == issue_num]
        if not issue_cases:
            print(f"Issue {issue_num}: no relevant cases, skipping\n")
            continue

        print(f"Issue {issue_num} — {len(issue_cases)} cases to extract")
        pdf_path = download_full_issue(issue_num)
        if not pdf_path:
            continue

        extracted = extract_cases_from_pdf(pdf_path, issue_cases)
        all_extracted.update(extracted)
        time.sleep(2)
        print()

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted, f, ensure_ascii=False, indent=2)

    print(f"{'─'*40}")
    print(f"Total cases extracted: {len(all_extracted)}")
    print(f"Saved → {output_path}")

if __name__ == "__main__":
    run()