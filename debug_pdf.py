import pdfplumber
import sys

def debug_pdf(pdf_path, num_pages=15):
    print(f"Reading: {pdf_path}\n")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:num_pages]):
            text = page.extract_text() or ""
            if text.strip():
                print(f"{'─'*40}")
                print(f"PAGE {i+1}")
                print(f"{'─'*40}")
                print(text[:600])
                print()

if __name__ == "__main__":
    debug_pdf("data/raw/full_issue_16.pdf", num_pages=15)