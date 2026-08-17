"""
READER - Azure AI Document Intelligence, `prebuilt-invoice` model.

This is the STRUCTURED EXTRACTOR half of the pairing, and the reason the two
approaches are meaningfully different rather than two flavours of the same
thing.

It is not a language model. It is a purpose-trained document model that
performs layout analysis and field detection, returning typed fields with
calibrated per-field confidence scores. It cannot hallucinate a plausible
invoice number, because it is not generating text - it is locating a region on
the page and reading it. Its failure modes are therefore completely different
from an LLM's: it fails by not finding a field, not by inventing one.

That is exactly the independence cross-validation needs.

Runs on the Azure FREE tier (F0), which supports prebuilt models on JPEG/PNG.
F0 is rate-limited to 1 transaction/second and 2 pages per document - both fine
for single-page invoices, but the runner must throttle to 1 worker for this
reader (see run.py).
"""

from __future__ import annotations

import os
import random
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

from fields import FIELDS, empty_record

MODEL = "prebuilt-invoice"
LABEL = "Azure DI"
MAX_RETRIES = 4

# Azure's invoice schema -> our 9 required fields.
#
# The mapping is one-to-one and unambiguous, which is the payoff of using a
# purpose-built invoice model: the concepts we need are first-class fields it
# was explicitly trained to find, not things we have to describe in a prompt
# and hope for.
FIELD_MAP = {
    "seller_name":    "VendorName",
    "seller_tax_id":  "VendorTaxId",
    "client_name":    "CustomerName",
    "client_tax_id":  "CustomerTaxId",
    "invoice_number": "InvoiceId",
    "invoice_date":   "InvoiceDate",
    "net_worth":      "SubTotal",
    "vat":            "TotalTax",
    "gross_worth":    "InvoiceTotal",
}


def _get(obj, name):
    """Read an attribute from either an SDK model object or a plain dict.

    The SDK returns typed models, but `AnalyzeResult` round-tripped through
    the response cache is a dict. Supporting both keeps the cache and the live
    path on identical code.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _field_value(field) -> str | None:
    """Pull a comparable string out of an Azure DocumentField.

    We prefer `content` - the raw text exactly as printed on the page - over
    the parsed typed value, because the LLM readers are instructed to
    transcribe verbatim. Comparing "as printed" against "as printed" keeps the
    readers answering the same question; the normalisation layer resolves the
    formatting differences afterwards.

    Falling back to the typed value covers the case where Azure recognised a
    field semantically but did not attach the source span.
    """
    if field is None:
        return None

    content = _get(field, "content")
    if content and str(content).strip():
        return str(content).strip()

    # Typed fallbacks, in the order Azure populates them.
    for attr in ("value_string", "valueString", "value_number", "valueNumber",
                 "value_date", "valueDate"):
        val = _get(field, attr)
        if val is not None and str(val).strip():
            return str(val).strip()

    currency = _get(field, "value_currency") or _get(field, "valueCurrency")
    if currency is not None:
        amount = _get(currency, "amount")
        if amount is not None:
            return str(amount)

    return None


def make_client():
    # Validate before constructing the client. An empty endpoint produces
    # "No connection adapters were found for '/documentintelligence/...'",
    # which sends you looking for a networking problem when the real cause is
    # a blank line in .env. Check the inputs and say so plainly.
    endpoint = os.environ.get("AZURE_DI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_DI_KEY", "").strip()

    if not endpoint:
        raise ValueError(
            "AZURE_DI_ENDPOINT is not configured in the runtime environment.")
    if not key:
        raise ValueError(
            "AZURE_DI_KEY is not configured in the runtime environment.")
    if not endpoint.startswith("http"):
        raise ValueError(
            f"AZURE_DI_ENDPOINT does not look like a URL: {endpoint!r}")

    return DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key))


def run(image_path: str, client=None) -> dict:
    """Extract the 9 fields from one invoice image. Never raises."""
    client = client or make_client()
    rec = empty_record()

    try:
        with open(image_path, "rb") as fh:
            image_bytes = fh.read()
    except Exception as exc:  # noqa: BLE001
        rec["_error"] = f"read: {type(exc).__name__}: {exc}"
        return rec

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            poller = client.begin_analyze_document(
                MODEL, AnalyzeDocumentRequest(bytes_source=image_bytes)
            )
            result = poller.result()

            documents = _get(result, "documents") or []
            if not documents:
                # Azure found no invoice at all. Nulls, not an exception - a
                # document it cannot recognise is a legitimate outcome that
                # the validation layer knows how to handle.
                rec["_error"] = "no invoice document detected"
                rec["_model"] = MODEL
                return rec

            fields = _get(documents[0], "fields") or {}
            confidences = {}

            for our_name, azure_name in FIELD_MAP.items():
                field = _get(fields, azure_name)
                rec[our_name] = _field_value(field)
                conf = _get(field, "confidence")
                if conf is not None:
                    confidences[our_name] = round(float(conf), 3)

            rec["_model"] = MODEL
            # Per-field confidence is something the LLM readers cannot
            # provide. It is a genuinely extra signal on any disputed field,
            # and worth carrying through to the report.
            rec["_confidence"] = confidences
            rec["_doc_confidence"] = _get(documents[0], "confidence")
            return rec

        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            msg = str(exc).lower()
            retryable = any(k in msg for k in
                            ("429", "rate", "quota", "throttl", "503", "500",
                             "timeout", "unavailable"))
            if retryable and attempt < MAX_RETRIES - 1:
                # F0 allows 1 transaction/second, so back off generously.
                time.sleep((2 ** attempt) + 1 + random.uniform(0, 1))
            else:
                break

    rec["_error"] = last_error
    return rec


def preflight(client) -> None:
    """Prove the endpoint and key work. Raises on failure.

    Azure requires images to be at least 50x50 pixels. A blank image at that
    minimum is enough because this checks credentials and connectivity, not
    extraction quality, and costs one page against the free allowance.
    """
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (50, 50), "white").save(buffer, format="PNG")
    poller = client.begin_analyze_document(
        MODEL, AnalyzeDocumentRequest(bytes_source=buffer.getvalue()))
    poller.result()
