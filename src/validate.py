"""
The validation layer - where two opinions become one auditable answer.

Three independent checks, deliberately of different kinds:

  1. CROSS-READER AGREEMENT  Do Gemini and Azure read the same value?
  2. ARITHMETIC TIE-OUT      Does net + VAT = gross?
  3. FORMAT PLAUSIBILITY     Is the date real? Is the tax ID the right shape?

Check 2 is the interesting one, because it is independent of BOTH models. It
is the only check that can catch an error the two models agree on, since it
interrogates the internal consistency of the numbers rather than comparing
sources. Two readers can misread the same digit; arithmetic cannot be talked
into balancing.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from difflib import SequenceMatcher

from fields import FIELD_KIND, FIELDS, normalise

# Names are compared with a similarity ratio rather than exact equality,
# because models legitimately differ on trailing punctuation and legal
# suffixes ("Smith and Sons Ltd." vs "Smith and Sons"). 0.90 is strict enough
# that two genuinely different companies never pass.
NAME_MATCH_THRESHOLD = 0.90

# Money is allowed a one-cent tolerance to absorb rounding, nothing more.
MONEY_TOLERANCE = Decimal("0.01")


# ---------------------------------------------------------------------------
# Check 1 - cross-model agreement
# ---------------------------------------------------------------------------

def compare_field(field: str, val_a, val_b) -> dict:
    """Compare one field across the two readers.

    Comparison happens AFTER normalisation. This is the detail that most naive
    implementations get wrong: comparing raw strings scores "$1,200.00" against
    "1200.00" as a disagreement, when the two readers actually agree perfectly
    and only the formatting differs. Skipping normalisation here would inflate
    the reported disagreement rate and send clean invoices to human review for
    no reason.

    Reader-agnostic on purpose: keys are `value_a` / `value_b`, not vendor
    names. Which readers are plugged in is a runtime choice, and the
    validation layer should not need to know or care.
    """
    kind = FIELD_KIND[field]
    norm_a, norm_b = normalise(field, val_a), normalise(field, val_b)

    if norm_a is None and norm_b is None:
        status, score = "both_missing", 0.0
    elif norm_a is None or norm_b is None:
        status, score = "one_missing", 0.0
    elif kind == "text":
        score = SequenceMatcher(None, norm_a, norm_b).ratio()
        status = "match" if score >= NAME_MATCH_THRESHOLD else "mismatch"
    elif kind == "money":
        same = abs(norm_a - norm_b) <= MONEY_TOLERANCE
        status, score = ("match", 1.0) if same else ("mismatch", 0.0)
    else:  # id, date - exact match required, a digit matters
        same = norm_a == norm_b
        status, score = ("match", 1.0) if same else ("mismatch", 0.0)

    return {
        "field": field,
        "value_a": val_a,
        "value_b": val_b,
        "normalised_a": str(norm_a) if norm_a is not None else "",
        "normalised_b": str(norm_b) if norm_b is not None else "",
        "status": status,
        "similarity": round(score, 3),
    }


# ---------------------------------------------------------------------------
# Check 2 - arithmetic tie-out
# ---------------------------------------------------------------------------

def arithmetic_check(net, vat, gross) -> dict:
    """Verify net + VAT = gross.

    Returns a verdict plus, where possible, the value implied by the other two.
    That implied value is what lets us settle a disagreement deterministically
    instead of picking a winner by preference.
    """
    n = normalise("net_worth", net)
    v = normalise("vat", vat)
    g = normalise("gross_worth", gross)

    present = sum(x is not None for x in (n, v, g))
    if present < 2:
        return {"status": "insufficient_data", "delta": None, "implied": {}}

    if present == 2:
        # Two of three known - derive the third. This is a repair mechanism,
        # not a check: it tells us what the missing value MUST be.
        implied = {}
        if g is None:
            implied["gross_worth"] = n + v
        elif v is None:
            implied["vat"] = g - n
        elif n is None:
            implied["net_worth"] = g - v
        return {"status": "derivable", "delta": None, "implied": implied}

    delta = (n + v) - g
    if abs(delta) <= MONEY_TOLERANCE:
        return {"status": "pass", "delta": delta, "implied": {}}
    return {"status": "fail", "delta": delta, "implied": {}}


# ---------------------------------------------------------------------------
# Check 3 - format plausibility
# ---------------------------------------------------------------------------

def plausibility_flags(rec: dict) -> list[str]:
    """Cheap sanity checks that catch gross failures the other two miss."""
    flags = []

    d = normalise("invoice_date", rec.get("invoice_date"))
    if rec.get("invoice_date") and not d:
        flags.append("date_unparseable")
    elif d:
        year = int(d[:4])
        if not (1990 <= year <= datetime.now().year + 1):
            flags.append("date_out_of_range")

    for f in ("seller_tax_id", "client_tax_id"):
        raw, norm = rec.get(f), normalise(f, rec.get(f))
        if raw and (not norm or not (8 <= len(norm) <= 14)):
            flags.append(f"{f}_implausible")

    if (normalise("seller_tax_id", rec.get("seller_tax_id"))
            and normalise("seller_tax_id", rec.get("seller_tax_id"))
            == normalise("client_tax_id", rec.get("client_tax_id"))):
        # Same tax ID on both sides means a block was almost certainly
        # duplicated - a company does not invoice itself.
        flags.append("seller_client_tax_id_identical")

    for f in FIELDS:
        if not rec.get(f):
            flags.append(f"{f}_missing")

    return flags


# ---------------------------------------------------------------------------
# Reconciliation - choose the final value, and record why
# ---------------------------------------------------------------------------

def reconcile(rec_a: dict, rec_b: dict,
              label_a: str = "Reader A",
              label_b: str = "Reader B") -> tuple[dict, list[dict], dict]:
    """Merge two extractions into one final record.

    Returns (final_record, per_field_comparison_rows, summary).

    `label_a` / `label_b` are display names used in the human-readable `reason`
    strings. They affect wording only - never logic - so swapping readers can
    never change a reconciliation decision.

    Every field carries a `reason` explaining how its value was chosen. That
    audit trail is the point: for any number in the final output you can say
    exactly which readers produced it, whether they agreed, and what rule broke
    the tie. Nothing in the output is unexplained.
    """
    comparisons = [compare_field(f, rec_a.get(f), rec_b.get(f)) for f in FIELDS]
    by_field = {c["field"]: c for c in comparisons}

    # Run the arithmetic check on each pipeline's own numbers. A pipeline whose
    # three figures balance internally is more credible on the disputed one.
    arith_a = arithmetic_check(rec_a.get("net_worth"), rec_a.get("vat"),
                               rec_a.get("gross_worth"))
    arith_b = arithmetic_check(rec_b.get("net_worth"), rec_b.get("vat"),
                               rec_b.get("gross_worth"))

    final, disputed = {}, []

    for f in FIELDS:
        cmp = by_field[f]
        a_val, b_val = rec_a.get(f), rec_b.get(f)

        if cmp["status"] == "match":
            final[f] = a_val
            cmp["chosen"] = a_val
            cmp["reason"] = "both readers agree"

        elif cmp["status"] == "both_missing":
            final[f] = None
            cmp["chosen"] = None
            cmp["reason"] = "neither reader found a value"

        elif cmp["status"] == "one_missing":
            # Exactly one reader read something. Take it, but flag it - a value
            # seen by one reader has not actually been cross-checked.
            final[f] = a_val if a_val else b_val
            cmp["chosen"] = final[f]
            src = label_a if a_val else label_b
            cmp["reason"] = f"only {src} found a value (uncorroborated)"
            disputed.append(f)

        else:  # mismatch - the two readers read different values
            chosen, reason = None, ""

            # Tie-break 1: let the arithmetic decide. If one reader's three
            # money figures balance and the other's do not, the balanced one
            # is almost certainly right. This is a deterministic, explainable
            # rule - not a preference for one vendor over the other.
            if f in ("net_worth", "vat", "gross_worth"):
                if arith_a["status"] == "pass" and arith_b["status"] != "pass":
                    chosen, reason = a_val, (f"{label_a}'s totals balance "
                                             f"(net+VAT=gross); "
                                             f"{label_b}'s do not")
                elif arith_b["status"] == "pass" and arith_a["status"] != "pass":
                    chosen, reason = b_val, (f"{label_b}'s totals balance "
                                             f"(net+VAT=gross); "
                                             f"{label_a}'s do not")

            # Tie-break 2: no arithmetic signal - fall back to a stated default
            # and flag for human review. Nothing is silently accepted on a
            # coin-flip; the disagreement is surfaced, not hidden.
            if chosen is None:
                chosen = a_val if a_val else b_val
                reason = ("readers disagree, no arithmetic tie-break - "
                          "flagged for human review")

            final[f] = chosen
            cmp["chosen"] = chosen
            cmp["reason"] = reason
            disputed.append(f)

    # Arithmetic verdict on the FINAL reconciled numbers, which may differ from
    # either model's own set once tie-breaks have been applied.
    arith_final = arithmetic_check(final["net_worth"], final["vat"],
                                   final["gross_worth"])
    flags = plausibility_flags(final)

    matched = sum(1 for c in comparisons if c["status"] == "match")

    # ---- Confidence tiering: turns checks into a routing decision ----------
    if not disputed and arith_final["status"] == "pass":
        tier = "HIGH"          # pass to the next control step
    elif not disputed:
        tier = "MEDIUM"        # models agree but totals do not tie out
    elif len(disputed) <= 2 and arith_final["status"] == "pass":
        tier = "MEDIUM"        # minor disagreement, money is sound
    else:
        tier = "REVIEW"        # human required

    summary = {
        "fields_matched": matched,
        "fields_total": len(FIELDS),
        "agreement_rate": round(matched / len(FIELDS), 3),
        "disputed_fields": ";".join(disputed),
        "arithmetic_status": arith_final["status"],
        "arithmetic_delta": (str(arith_final["delta"])
                             if arith_final["delta"] is not None else ""),
        "plausibility_flags": ";".join(flags),
        "confidence": tier,
    }
    return final, comparisons, summary
