"""
Quick check — does gemini-2.5-flash-lite have a usable quota on this
account, separate from the 20/day cap hit on gemini-2.5-flash?
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
from google import genai
from google.genai import types as genai_types
client = genai.Client(api_key=api_key)

print("Testing gemini-2.5-flash-lite ...")
try:
    response = client.models.generate_content(
        model    = "gemini-2.5-flash-lite",
        contents = "Reply with exactly: {\"status\": \"ok\"}",
        config   = genai_types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json",
        ),
    )
    print("SUCCESS:", response.text.strip())
except Exception as exc:
    print(f"FAIL: {exc}")
