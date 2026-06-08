"""
List all Gemini models available for your API key.
Run this once to find the right model name.

Usage:
    python scripts/list_models.py
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
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

from google import genai
client = genai.Client(api_key=api_key)

print("Available models that support generateContent:\n")
found = []
for m in client.models.list():
    name    = m.name
    methods = getattr(m, "supported_actions", None) or []
    # Filter to generative models only
    if any(x in name.lower() for x in ["gemini", "flash", "pro"]):
        print(f"  {name}")
        found.append(name)

print(f"\nTotal: {len(found)} models")
print("\nCopy one of the names above and paste it back.")
