"""
The extraction contract, shared by both pipelines.

Two things live here, and it matters that they are shared:

  1. InvoiceExtraction - the Pydantic model that becomes the JSON Schema we
     hand to BOTH providers as a structured-output / tool-use spec.
  2. EXTRACTION_PROMPT - the identical instruction sent to both models.

Keeping the contract in one place prevents provider integrations from silently
drifting to different field definitions. Gemini uses the schema directly;
Azure maps its trained invoice fields to the same names before comparison.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InvoiceExtraction(BaseModel):
    """The 9 required fields. Every field is Optional by design.

    A model that cannot find a field must return null. Forcing a value would
    guarantee hallucination on the fields that are genuinely absent, and a
    plausible wrong number is far more dangerous downstream than a visible
    blank - a blank stops the process, a wrong number silently enters a ledger.
    """

    seller_name: Optional[str] = Field(
        description="Legal name of the company issuing the invoice (the Seller). "
                    "Name only - exclude street address, city, postcode."
    )
    seller_tax_id: Optional[str] = Field(
        description="Tax ID / VAT number of the SELLER, exactly as printed "
                    "including any hyphens."
    )
    client_name: Optional[str] = Field(
        description="Legal name of the company being billed (the Client/Buyer). "
                    "Name only - exclude street address, city, postcode."
    )
    client_tax_id: Optional[str] = Field(
        description="Tax ID / VAT number of the CLIENT, exactly as printed "
                    "including any hyphens."
    )
    invoice_number: Optional[str] = Field(
        description="The invoice number or reference, exactly as printed."
    )
    invoice_date: Optional[str] = Field(
        description="The invoice issue date, transcribed EXACTLY as printed on "
                    "the document. Do not reformat or convert it."
    )
    net_worth: Optional[str] = Field(
        description="The TOTAL net worth / subtotal before tax, from the summary "
                    "or total row - not an individual line item. Digits and "
                    "decimal separator only, no currency symbol."
    )
    vat: Optional[str] = Field(
        description="The TOTAL VAT / tax amount from the summary or total row - "
                    "not a percentage, not a line item. Digits and decimal "
                    "separator only, no currency symbol."
    )
    gross_worth: Optional[str] = Field(
        description="The TOTAL gross worth / grand total including tax, from the "
                    "summary or total row. Digits and decimal separator only, "
                    "no currency symbol."
    )


# ---------------------------------------------------------------------------
# The shared prompt
# ---------------------------------------------------------------------------
# Every constraint below exists to close a specific failure mode observed in
# document-extraction work. The comments name which one.

EXTRACTION_PROMPT = """You are a precise document-extraction system reading a commercial invoice.

Extract the nine required fields into the provided schema.

RULES:

1. TRANSCRIBE, DO NOT INTERPRET.
   Return characters exactly as printed on the document. Do not reformat dates.
   Do not add or remove currency symbols beyond what the schema asks for. Do not
   normalise names or expand abbreviations.

2. NEVER CALCULATE.
   Do not compute net_worth, vat, or gross_worth from the line items, and do not
   derive one from the other two. Read each of the three directly from the
   summary/total row. If a value is not printed, return null for it.

3. NULL IS A VALID AND CORRECT ANSWER.
   If a field is not clearly legible or not present, return null. Do not guess,
   do not infer from context, do not substitute a similar-looking value from
   elsewhere on the page. A null is useful; a confident wrong value is harmful.

4. SELLER vs CLIENT.
   Invoices place these in two visually similar blocks, each with a name and a
   tax ID. Use the block headings and page layout to decide which is which.
   The SELLER issues the invoice; the CLIENT is billed. Do not swap them, and do
   not copy one block's tax ID into the other's field.

5. TOTALS, NOT LINE ITEMS.
   Where an invoice has a per-item table and a summary row, always take the
   summary/total figures. The largest gross figure in the table is usually a line
   item, not the total - read the labelled total row.

Return only the structured data."""
