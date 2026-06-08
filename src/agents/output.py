"""
BanglaLex — Agent 4: Output Agent
====================================
Client-facing output layer.

Responsibility
--------------
Translate the legal analysis and judgment prediction into clear,
actionable plain-language guidance — in the SAME LANGUAGE the client
used to describe their problem (English or Bangla).

This maps directly to Figure 7.2 in the thesis proposal:
  • Probability-based outcome statement
  • Plain-language reasoning (no legal jargon)
  • Concrete next steps
  • Evidence to collect
  • Urgency note (if applicable)

Input  : analysis dict from Agent 3
Output : output dict with keys —
           language, probability_statement, plain_reasoning,
           next_steps, evidence_to_collect, urgency_note
"""

import logging
from .base import GeminiAgent

logger = logging.getLogger(__name__)

# ── System prompts (one per language) ─────────────────────────────────────────

_SYSTEM_EN = """
You are a plain-language legal guide for BanglaLex, an AI legal assistance
system for Bangladesh. Your role is to explain complex legal analysis in simple,
everyday English that any ordinary citizen can understand.

Avoid legal jargon. Write as if explaining to a friend, not a lawyer.

Return a JSON object with exactly these keys:

{
  "probability_statement": "One sentence stating the likely outcome in plain language with a percentage. Example: 'Based on the law and similar cases, there is approximately a 70% chance of a favourable ruling in your case.'",
  "plain_reasoning": "3-4 sentences explaining WHY in simple language. Begin by citing the specific Act and Section (e.g., 'Under Section 17 of the Cantonments Rent Restriction Act, 1963...'), then explain in plain words what that law says and exactly why it applies to this case.",
  "next_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ..."
  ],
  "evidence_to_collect": ["document or evidence 1 to gather", "document 2", ...],
  "urgency_note": "A time-sensitive warning if any deadline or immediate action is needed, or null if not urgent."
}

Return ONLY the JSON object.
""".strip()

_SYSTEM_BN = """
আপনি BanglaLex-এর একজন সরল ভাষার আইনি গাইড। আপনার কাজ হল জটিল আইনি বিশ্লেষণকে
সহজ, দৈনন্দিন বাংলায় ব্যাখ্যা করা যা যেকোনো সাধারণ নাগরিক বুঝতে পারে।

আইনি পরিভাষা এড়িয়ে চলুন। এমনভাবে লিখুন যেন একজন বন্ধুকে বোঝাচ্ছেন।

নিচের কী-সহ একটি JSON অবজেক্ট রিটার্ন করুন:

{
  "probability_statement": "একটি বাক্যে সম্ভাব্য ফলাফল সরল ভাষায় শতাংশসহ বলুন। উদাহরণ: 'আইন ও একই ধরনের মামলার ভিত্তিতে, আপনার মামলায় অনুকূল রায়ের সম্ভাবনা প্রায় ৭০%।'",
  "plain_reasoning": "৩-৪ বাক্যে কেন এই সম্ভাবনা তা ব্যাখ্যা করুন। প্রথমে নির্দিষ্ট আইনের নাম ও ধারা নম্বর উল্লেখ করুন (যেমন: 'ক্যান্টনমেন্ট ভাড়া নিয়ন্ত্রণ আইন, ১৯৬৩-এর ১৭ ধারা অনুযায়ী...'), তারপর সহজ ভাষায় বলুন আইনটি কী বলে এবং কেন এই মামলায় প্রযোজ্য।",
  "next_steps": [
    "পদক্ষেপ ১: ...",
    "পদক্ষেপ ২: ...",
    "পদক্ষেপ ৩: ...",
    "পদক্ষেপ ৪: ..."
  ],
  "evidence_to_collect": ["সংগ্রহ করার প্রমাণ বা দলিল ১", "দলিল ২", ...],
  "urgency_note": "কোনো সময়সীমা বা জরুরি পদক্ষেপ থাকলে সতর্কতামূলক বার্তা, অথবা জরুরি না হলে null।"
}

শুধুমাত্র JSON অবজেক্টটি রিটার্ন করুন।
""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _confidence_to_pct(confidence: float) -> int:
    """Convert 0–1 confidence to a rounded percentage."""
    return round(confidence * 100 / 5) * 5    # round to nearest 5%


def _format_analysis(analysis: dict) -> str:
    """Summarise the full analysis dict for the output prompt."""
    outcome    = analysis.get("predicted_outcome", "uncertain")
    confidence = _confidence_to_pct(analysis.get("confidence", 0.5))
    reasoning  = analysis.get("reasoning_chain", "")
    supporting = analysis.get("supporting_factors", [])
    risks      = analysis.get("risk_factors", [])
    statutes   = analysis.get("applicable_statutes", [])
    urgency    = analysis.get("urgency", "medium")

    top_statute = ""
    if statutes:
        s = statutes[0]
        top_statute = (
            f"{s.get('act_name','')}"
            + (f" §{s.get('section_no','')}" if s.get('section_no') else "")
            + (f": {s.get('relevance','')}" if s.get('relevance') else "")
        )

    return (
        f"Predicted outcome : {outcome.upper()} ({confidence}% confidence)\n"
        f"Case urgency      : {urgency}\n"
        f"Key statute       : {top_statute or 'see below'}\n\n"
        f"Legal reasoning   :\n{reasoning}\n\n"
        f"Supporting factors: {'; '.join(supporting) or 'none listed'}\n"
        f"Risk factors      : {'; '.join(risks) or 'none listed'}\n"
        f"Recommended basis : {analysis.get('recommended_legal_basis','')}"
    )


# ── Agent class ────────────────────────────────────────────────────────────────

class OutputAgent(GeminiAgent):
    """
    Agent 4 — Output Agent.

    Produces a plain-language explanation in the client's language.

    Parameters
    ----------
    model_name : Gemini model
    """

    def __init__(self, model_name: str = None):
        super().__init__(model_name=model_name, temperature=0.3)

    def generate(self, analysis: dict) -> dict:
        """
        Generate a client-facing plain-language explanation.

        Parameters
        ----------
        analysis : output of LegalJudgmentAgent.predict()

        Returns
        -------
        dict with keys: language, probability_statement, plain_reasoning,
                        next_steps, evidence_to_collect, urgency_note
        """
        logger.info("Agent 4 — generating client output …")

        language    = analysis.get("language", "en")
        system_p    = _SYSTEM_BN if language == "bn" else _SYSTEM_EN
        confidence  = _confidence_to_pct(analysis.get("confidence", 0.5))
        outcome     = analysis.get("predicted_outcome", "uncertain")

        prompt = f"""{system_p}

---
CASE ANALYSIS SUMMARY:
{_format_analysis(analysis)}

Predicted outcome: {outcome} ({confidence}% confidence)
Client language  : {"Bangla (bn)" if language == "bn" else "English (en)"}
---

Generate the client-facing explanation now."""

        output = self._call_json(prompt)

        output["language"] = language

        # Ensure all keys are present
        output.setdefault("probability_statement", "")
        output.setdefault("plain_reasoning",       "")
        output.setdefault("next_steps",            [])
        output.setdefault("evidence_to_collect",   [])
        output.setdefault("urgency_note",          None)

        logger.info("Agent 4 done")
        return output