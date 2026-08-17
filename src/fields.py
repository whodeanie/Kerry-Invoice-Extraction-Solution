"""
Shared field definitions and normalisation.

Both pipelines produce a dict with exactly these 9 keys. Everything downstream
(comparison, validation, CSV writing) depends only on this contract, which is
what lets the two pipelines stay completely independent of each other.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

# The 9 required fields, in the order they appear in the brief.
FIELDS = [
    "seller_name",
    "seller_tax_id",
    "client_name",
    "client_tax_id",
    "invoice_number",
    "invoice_date",
    "net_worth",
    "vat",
    "gross_worth",
]

# How each field should be compared. This drives the whole validation layer.
#   "text"  -> fuzzy compare (OCR drops punctuation, casing varies)
#   "id"    -> exact compare on digits only
#   "money" -> exact compare as Decimal
#   "date"  -> exact compare as ISO date
FIELD_KIND = {
    "seller_name": "text",
    "seller_tax_id": "id",
    "client_name": "text",
    "client_tax_id": "id",
    "invoice_number": "id",
    "invoice_date": "date",
    "net_worth": "money",
    "vat": "money",
    "gross_worth": "money",
}

MONEY_FIELDS = [f for f, k in FIELD_KIND.items() if k == "money"]


def empty_record() -> dict:
    """A record with every field present but unset. Pipelines start from this
    so a crash mid-extraction still yields a well-shaped row."""
    return {f: None for f in FIELDS}


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
# This is the layer that stops us reporting false disagreements. If Pipeline A
# returns "$1,200.00" and Pipeline B returns "1200", the pipelines agree - only
# the formatting differs. Comparing raw strings would score that as a mismatch
# and quietly inflate the error rate.


def norm_money(value) -> Decimal | None:
    """'$ 1,234.56' / '1234.56' / '1 234,56' -> Decimal('1234.56')."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Drop currency symbols, spaces, and any stray letters.
    s = re.sub(r"[^\d.,\-]", "", s)
    if not s:
        return None

    # Decide which separator is the decimal point. Whichever appears LAST is the
    # decimal separator; the other is a thousands separator. This handles both
    # the 1,234.56 (US) and 1.234,56 (EU) conventions without a locale flag.
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot > last_comma:
        s = s.replace(",", "")
    elif last_comma > last_dot:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")

    try:
        return Decimal(s).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


# Date formats seen in the wild on this dataset, most specific first.
_DATE_FORMATS = [
    "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%d.%m.%Y", "%m.%d.%Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y",
    "%d %b %Y", "%Y/%m/%d",
]


def norm_date(value) -> str | None:
    """Any recognised date format -> 'YYYY-MM-DD'.

    Note the m/d vs d/m ambiguity: '01/03/2016' is genuinely undecidable in
    isolation. We try US ordering first because this dataset is US-formatted.
    Where the day is > 12 the format is self-disambiguating and we get it right
    regardless of ordering. This ambiguity is worth naming out loud rather than
    pretending it is solved.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"\s+", " ", s)
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if 1990 <= dt.year <= 2100:  # plausibility guard
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def norm_id(value) -> str | None:
    """Tax IDs and invoice numbers -> digits only.

    '958-74-3462' and '958743462' are the same identifier; the hyphens are
    display formatting. Stripping them removes a whole class of false mismatches.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


# ONLY genuine legal-form suffixes belong here.
#
# An earlier version also folded "group", "sons" and "and sons". That was a
# real bug, and an instructive one: those words are IDENTIFYING parts of a
# company name, not legal forms. Folding them turned "Johnson Group Inc" into
# "johnson", so "Johnson Group" and "Johnson Inc" compared as equal.
#
# Over-normalisation is as dangerous as under-normalisation, and it is worse in
# one specific way: under-normalisation creates false MISMATCHES, which surface
# as extra human review. Over-normalisation creates false MATCHES - two
# different suppliers agreeing - which surface as nothing at all. The system
# reports confident agreement on an invoice it got wrong.
#
# When in doubt, do not fold. A false mismatch costs a minute of review; a
# false match can pay the wrong company.
_LEGAL_SUFFIXES = (r"\b(inc|incorporated|llc|llp|ltd|limited|corp|corporation"
                   r"|plc|gmbh|ag|sa|nv|bv|pty|co)\b")

# OCR drops spaces, so a suffix can arrive glued to the preceding word
# ("SonsLtd"). Un-glue it before folding, otherwise the \b boundary never
# matches and the suffix survives on one side of the comparison but not the
# other - producing a mismatch between two identical names.
_GLUED_SUFFIX = re.compile(
    r"([a-z])(inc|llc|ltd|limited|corp|plc|gmbh|pty)\b", re.I)


def norm_text(value) -> str | None:
    """Company/person names -> lowercase, punctuation and legal suffixes folded.

    OCR reliably mangles trailing punctuation ('Ltd.' vs 'Ltd'), and the legal
    form carries no identifying information, so we remove both before
    comparing. We keep the ORIGINAL string for output - this normalised form is
    only ever used for the match decision.
    """
    if value is None:
        return None
    s = str(value).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)          # punctuation -> space
    s = _GLUED_SUFFIX.sub(r"\1 \2", s)      # "sonsltd" -> "sons ltd"
    s = re.sub(_LEGAL_SUFFIXES, " ", s)     # fold legal forms only
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalise(field: str, value):
    """Dispatch to the right normaliser for a field."""
    kind = FIELD_KIND.get(field, "text")
    return {
        "money": norm_money,
        "date": norm_date,
        "id": norm_id,
        "text": norm_text,
    }[kind](value)


def normalise_record(rec: dict) -> dict:
    """Normalise every field in a record."""
    return {f: normalise(f, rec.get(f)) for f in FIELDS}
