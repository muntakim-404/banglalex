import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import transformers
print(f"Transformers: {transformers.__version__}")

import faiss
print(f"FAISS: loaded OK")

import langchain
print(f"LangChain: {langchain.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

import pdfplumber
print(f"pdfplumber: loaded OK")

import pytesseract
import config
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
print(f"Tesseract: {pytesseract.get_tesseract_version()}")

print("\n All good — environment is ready.")