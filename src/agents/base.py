"""
BanglaLex — Base Agent (Multi-backend: Groq + Gemini)
========================================================
Shared infrastructure used by all four agents.

Backend is auto-detected from the model_name string:
  - contains "gemini"  -> Google Gemini API (google-genai SDK)
  - anything else      -> Groq API

This lets every existing call site (judgment.py, evaluator.py,
baseline.py, phase4_evaluate.py --model flag) work unchanged —
just pass a different model_name string to switch backbones.

Environment
-----------
Add to your .env file:
    GROQ_API_KEY=your_groq_key_here
    GEMINI_API_KEY=your_gemini_key_here

Get a free Groq key   : https://console.groq.com/keys
Get a free Gemini key : https://aistudio.google.com/apikey
  (Create it from AI Studio directly — NOT Google Cloud Console —
   AI Studio keys default to the free tier with no billing required.)
"""

import os
import re
import json
import time
import logging

logger = logging.getLogger(__name__)


# ── Language detection ──────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Return "bn" if text is primarily Bangla, otherwise "en".
    Bangla Unicode block: U+0980 – U+09FF.
    Threshold: > 30% of alphabetic characters are Bangla.
    """
    bangla = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    total  = sum(1 for c in text if c.isalpha())
    return "bn" if (total > 0 and bangla / total > 0.3) else "en"


# ── Base agent class (multi-backend) ────────────────────────────────────

class GeminiAgent:
    """
    Base class for all BanglaLex agents.
    Backend (Groq vs Gemini) is auto-detected from model_name.

    Default model : meta-llama/llama-4-scout-17b-16e-instruct  (Groq)
      -> Free tier  : 500,000 tokens/day
      -> Multilingual: English + Bangla supported
      -> JSON mode  : supported

    To use Gemini instead, pass any model_name containing "gemini",
    e.g. model_name="gemini-2.5-flash" or "gemini-2.5-flash-lite".
      -> Free tier (Flash)      : 10 RPM / 250 RPD / 250K TPM
      -> Free tier (Flash-Lite) : 15 RPM / 1,000 RPD / 250K TPM
      -> Avoid gemini-2.5-pro: only 5 RPM / 100 RPD on free tier.

    Parameters
    ----------
    model_name  : model identifier (determines backend)
    temperature : 0.0–1.0; low = more deterministic
    """

    DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(
        self,
        model_name:  str   = None,
        temperature: float = 0.1,
    ):
        # Load .env file if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self.model_name  = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self.backend      = "gemini" if "gemini" in self.model_name.lower() else "groq"

        if self.backend == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set.\n"
                    "Add this line to your .env file:\n"
                    "    GROQ_API_KEY=your_key_here\n"
                    "Get a free key at: https://console.groq.com/keys"
                )
            from groq import Groq
            self._client = Groq(api_key=api_key)

        else:  # gemini
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set.\n"
                    "Add this line to your .env file:\n"
                    "    GEMINI_API_KEY=your_key_here\n"
                    "Get a free key at: https://aistudio.google.com/apikey\n"
                    "(Create it from AI Studio, not Google Cloud Console.)"
                )
            from google import genai
            from google.genai import types as genai_types
            self._client = genai.Client(api_key=api_key)
            self._types  = genai_types

        logger.info(
            f"{self.__class__.__name__} ready | backend={self.backend} | "
            f"model={self.model_name}"
        )

    # ── Internal call helpers ───────────────────────────────────────────

    def _call_json(self, prompt: str, retries: int = 2) -> dict:
        """
        Call the LLM and parse the response as JSON.
        Retries with back-off on transient errors (rate limits, timeouts).
        """
        last_err = None

        for attempt in range(retries + 1):
            try:
                if self.backend == "groq":
                    response = self._client.chat.completions.create(
                        model           = self.model_name,
                        messages        = [{"role": "user", "content": prompt}],
                        temperature     = self.temperature,
                        response_format = {"type": "json_object"},
                    )
                    text = response.choices[0].message.content.strip()

                else:  # gemini
                    response = self._client.models.generate_content(
                        model    = self.model_name,
                        contents = prompt,
                        config   = self._types.GenerateContentConfig(
                            temperature        = self.temperature,
                            response_mime_type = "application/json",
                        ),
                    )
                    text = response.text.strip()

                # Defensive strip in case the model wraps output in markdown fences
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$",           "", text).strip()
                return json.loads(text)

            except Exception as exc:
                last_err = exc
                # Gemini free tier (10-15 RPM) needs a longer minimum wait than Groq
                base_wait = 8 if self.backend == "gemini" else 5
                wait = base_wait * (2 ** attempt)
                logger.warning(
                    f"{self.__class__.__name__}._call_json attempt "
                    f"{attempt + 1}/{retries + 1} failed: "
                    f"{str(exc)[:160]}"
                )
                if attempt < retries:
                    logger.info(f"  Waiting {wait}s before retry …")
                    time.sleep(wait)

        raise RuntimeError(
            f"{self.__class__.__name__} JSON call failed after "
            f"{retries + 1} attempts: {last_err}"
        )

    def _call_text(self, prompt: str) -> str:
        """Call the LLM and return the raw text response."""
        if self.backend == "groq":
            response = self._client.chat.completions.create(
                model       = self.model_name,
                messages    = [{"role": "user", "content": prompt}],
                temperature = self.temperature,
            )
            return response.choices[0].message.content.strip()

        else:  # gemini
            response = self._client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
                config   = self._types.GenerateContentConfig(
                    temperature = self.temperature,
                ),
            )
            return response.text.strip()
