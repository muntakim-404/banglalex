"""
BanglaLex — Base Agent (Groq backend)
=======================================
Shared infrastructure used by all four agents.
Uses the Groq API with llama-3.3-70b-versatile.

Environment
-----------
Add to your .env file:
    GROQ_API_KEY=your_key_here

Get a free key at: https://console.groq.com/keys
"""

import os
import re
import json
import time
import logging

logger = logging.getLogger(__name__)


# ── Language detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Return "bn" if text is primarily Bangla, otherwise "en".
    Bangla Unicode block: U+0980 – U+09FF.
    Threshold: > 30% of alphabetic characters are Bangla.
    """
    bangla = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    total  = sum(1 for c in text if c.isalpha())
    return "bn" if (total > 0 and bangla / total > 0.3) else "en"


# ── Groq base class ────────────────────────────────────────────────────────────

class GeminiAgent:
    """
    Base class for all BanglaLex agents.
    Named GeminiAgent for compatibility but now backed by Groq.

    Default model : llama-3.3-70b-versatile
      → Free tier  : 6,000 requests/day, 500,000 tokens/min
      → Multilingual: English + Bangla supported
      → JSON mode  : supported

    Parameters
    ----------
    model_name  : Groq model identifier
    temperature : 0.0–1.0; low = more deterministic
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

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

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Add this line to your .env file:\n"
                "    GROQ_API_KEY=your_key_here\n"
                "Get a free key at: https://console.groq.com/keys"
            )

        from groq import Groq
        self.model_name  = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self._client     = Groq(api_key=api_key)

        logger.info(f"{self.__class__.__name__} ready | model={self.model_name}")

    # ── Internal call helpers ──────────────────────────────────────────────────

    def _call_json(self, prompt: str, retries: int = 2) -> dict:
        """
        Call Groq and parse the response as JSON.
        Uses Groq's JSON mode for reliable structured output.
        Retries with back-off on transient errors.
        """
        last_err = None

        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model           = self.model_name,
                    messages        = [{"role": "user", "content": prompt}],
                    temperature     = self.temperature,
                    response_format = {"type": "json_object"},
                )
                text = response.choices[0].message.content.strip()
                # Defensive strip in case model wraps in markdown fences
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$",           "", text).strip()
                return json.loads(text)

            except Exception as exc:
                last_err = exc
                wait = 5 * (2 ** attempt)   # 5s → 10s back-off
                logger.warning(
                    f"{self.__class__.__name__}._call_json attempt "
                    f"{attempt + 1}/{retries + 1} failed: "
                    f"{str(exc)[:120]}"
                )
                if attempt < retries:
                    logger.info(f"  Waiting {wait}s before retry …")
                    time.sleep(wait)

        raise RuntimeError(
            f"{self.__class__.__name__} JSON call failed after "
            f"{retries + 1} attempts: {last_err}"
        )

    def _call_text(self, prompt: str) -> str:
        """Call Groq and return the raw text response."""
        response = self._client.chat.completions.create(
            model       = self.model_name,
            messages    = [{"role": "user", "content": prompt}],
            temperature = self.temperature,
        )
        return response.choices[0].message.content.strip()
