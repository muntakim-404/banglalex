import os

# Tesseract path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# LLM — using Groq API (free, no GPU required for inference)
LLM_PROVIDER = "groq"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = "llama3-70b-8192"

# Embedding model — lightweight, fits your 4GB VRAM
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")
STATUTES_DIR = os.path.join(DATA_DIR, "statutes")