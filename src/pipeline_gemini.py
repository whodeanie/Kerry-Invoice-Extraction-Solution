"""
READER - Google Gemini vision extraction.

The LLM half of the pairing. Sends the invoice image to a Gemini Flash model
and constrains the response with `response_schema`, so the reply is validated
JSON rather than prose we have to parse.

Runs on Google's FREE tier, which covers Flash models with image input. That
matters practically: the whole 50-image run costs nothing.
"""

from __future__ import annotations

import json
import mimetypes
import os
import random
import time

from google import genai
from google.genai import types

from schema import EXTRACTION_PROMPT, InvoiceExtraction
from fields import FIELDS, empty_record

# Keep the extraction approach fixed to Gemini Flash while accommodating the
# free tier's per-model daily request cap. Models are tried in this explicit,
# reproducible order, and the exact model used is recorded in every raw result.
MODEL_POOL = (
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)
MODEL = "Gemini Flash pinned pool"
_active_model_index = 0

MAX_RETRIES = 4
LABEL = "Gemini"

# Null-ish strings models emit instead of a real null. Normalised back to None
# so the comparison layer sees a genuine missing value rather than the literal
# text "N/A" being treated as data.
NULLISH = {"", "null", "none", "n/a", "na", "not visible",
           "not present", "unknown", "not found"}


def _clean(value):
    if isinstance(value, str) and value.strip().lower() in NULLISH:
        return None
    return value


def run(image_path: str, client=None) -> dict:
    """Extract the 9 fields from one invoice image. Never raises."""
    global _active_model_index
    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rec = empty_record()

    mime, _ = mimetypes.guess_type(image_path)
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/heic"):
        mime = "image/jpeg"

    try:
        with open(image_path, "rb") as fh:
            image_bytes = fh.read()
    except Exception as exc:  # noqa: BLE001
        rec["_error"] = f"read: {type(exc).__name__}: {exc}"
        return rec

    last_error = None

    while _active_model_index < len(MODEL_POOL):
        model = MODEL_POOL[_active_model_index]
        daily_quota_exhausted = False

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        EXTRACTION_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,               # maximum determinism
                        response_mime_type="application/json",
                        # The same Pydantic model both other readers use, so all
                        # readers are provably answering the identical question.
                        response_schema=InvoiceExtraction,
                    ),
                )

                data = json.loads(resp.text)
                for f in FIELDS:
                    rec[f] = _clean(data.get(f))
                rec["_model"] = model
                usage = getattr(resp, "usage_metadata", None)
                rec["_tokens"] = getattr(usage, "total_token_count", None)
                return rec

            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                msg = str(exc).lower()
                daily_quota_exhausted = any(k in msg for k in (
                    "generate_content_free_tier_requests",
                    "generaterequestsperday",
                ))
                if daily_quota_exhausted:
                    break

                # Retry only transient failures. Invalid credentials and bad
                # requests fail identically on every attempt.
                retryable = any(k in msg for k in
                                ("429", "rate", "503", "500", "timeout",
                                 "unavailable", "deadline"))
                if retryable and attempt < MAX_RETRIES - 1:
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                else:
                    break

        if daily_quota_exhausted:
            _active_model_index += 1
            continue
        break

    rec["_error"] = last_error
    return rec


def make_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def preflight(client) -> None:
    """Verify API access without consuming a generation request."""
    client.models.get(model=MODEL_POOL[0])
