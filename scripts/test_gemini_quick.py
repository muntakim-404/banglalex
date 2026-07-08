"""
Quick sanity check — run this BEFORE the full 90-case Gemini evaluation.
Confirms your GEMINI_API_KEY actually works and isn't hitting the
account-level limit:0 issue from before.

Usage:
    python scripts/test_gemini_quick.py
"""
import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("FAIL: GEMINI_API_KEY not set in .env")
    sys.exit(1)

from google import genai
from google.genai import types as genai_types

client = genai.Client(api_key=api_key)

print("Sending one test request to gemini-2.5-flash ...")
try:
    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = "Reply with exactly the JSON object: {\"status\": \"ok\"}",
        config   = genai_types.GenerateContentConfig(
            temperature        = 0.0,
            response_mime_type = "application/json",
        ),
    )
    print("\nSUCCESS — response received:")
    print(response.text.strip())
    print("\nUsage metadata:", response.usage_metadata)
    print("\nYour Gemini key works. You're clear to run the full evaluation.")
except Exception as exc:
    print(f"\nFAIL: {type(exc).__name__}: {exc}")
    print("\nIf this says 'limit: 0' again — same as last time — it's an")
    print("account/project issue, not a model-name issue. Things to check:")
    print("  1. Was this key created at https://aistudio.google.com/apikey")
    print("     directly (NOT via Google Cloud Console)?")
    print("  2. Try creating a brand new key with a brand new Google account")
    print("     to rule out a project-level restriction.")
    print("  3. Check console output for any region-based restriction notice.")
    sys.exit(1)
