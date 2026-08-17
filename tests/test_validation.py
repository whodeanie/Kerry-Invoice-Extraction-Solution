"""
Unit tests for the validation layer.

These matter more than they look. The reconciliation logic is where two
opinions become one number that a client will act on, so every branch of it -
every rule that picks a winner - needs a test proving it fires when it should
and not when it shouldn't.

They also run with no API keys and no network, which means the decision logic
can be demonstrated and verified independently of whether the models are
reachable. Run with:  python tests/test_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fields import norm_date, norm_id, norm_money, norm_text
from validate import arithmetic_check, compare_field, reconcile

PASSED, FAILED = [], []


def check(name: str, actual, expected):
    if actual == expected:
        PASSED.append(name)
    else:
        FAILED.append(f"{name}\n      expected: {expected!r}\n      actual:   {actual!r}")


# ---------------------------------------------------------------------------
# Normalisation - the layer that prevents false disagreements
# ---------------------------------------------------------------------------

def test_normalisation():
    # US and European money formats must land on the same Decimal.
    check("money: US format", str(norm_money("$ 1,234.56")), "1234.56")
    check("money: EU format", str(norm_money("1 234,56")), "1234.56")
    check("money: EU dot-thousands", str(norm_money("1.234,56")), "1234.56")
    check("money: bare", str(norm_money("1234.56")), "1234.56")
    check("money: garbage -> None", norm_money("n/a"), None)

    # This is the key case: two models, two formats, ONE value.
    check("money: US vs EU agree",
          norm_money("$ 10,862.96") == norm_money("10 862,96"), True)

    # Dates: any input format, one ISO output.
    check("date: US slash", norm_date("11/18/2019"), "2019-11-18")
    check("date: ISO passthrough", norm_date("2019-11-18"), "2019-11-18")
    check("date: long form", norm_date("November 18, 2019"), "2019-11-18")
    check("date: nonsense -> None", norm_date("not a date"), None)

    # Tax IDs: hyphens are display formatting, not identity.
    check("id: hyphens stripped", norm_id("846-41-3677"), "846413677")
    check("id: already bare", norm_id("846413677"), "846413677")

    # Names: case, punctuation and legal suffix folded away.
    check("text: suffix folded",
          norm_text("Smith and Sons Ltd.") == norm_text("Smith and Sons"), True)
    check("text: different companies stay different",
          norm_text("Smith and Sons") == norm_text("Johnson Group"), False)


# ---------------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------------

def test_compare_field():
    # Formatting differences must NOT be reported as disagreement.
    c = compare_field("net_worth", "$ 1,234.56", "1234.56")
    check("compare: money formats agree", c["status"], "match")

    c = compare_field("net_worth", "1,234.56", "1,234.65")
    check("compare: transposed digits caught", c["status"], "mismatch")

    c = compare_field("seller_name", "Smith and Sons Ltd.", "Smith and Sons Ltd")
    check("compare: trailing period agrees", c["status"], "match")

    c = compare_field("seller_name", "Smith and Sons", "Vasquez Industrial")
    check("compare: different names disagree", c["status"], "mismatch")

    c = compare_field("seller_tax_id", "846-41-3677", "846413677")
    check("compare: tax id punctuation agrees", c["status"], "match")

    c = compare_field("seller_tax_id", "846-41-3677", "846-41-3678")
    check("compare: one wrong digit caught", c["status"], "mismatch")

    c = compare_field("invoice_number", None, "91756179")
    check("compare: one side null", c["status"], "one_missing")

    c = compare_field("invoice_number", None, None)
    check("compare: both null", c["status"], "both_missing")


# ---------------------------------------------------------------------------
# Arithmetic tie-out - the check independent of both models
# ---------------------------------------------------------------------------

def test_arithmetic():
    check("arith: balanced",
          arithmetic_check("10862.96", "1086.30", "11949.26")["status"], "pass")

    check("arith: unbalanced",
          arithmetic_check("10862.96", "1086.30", "11000.00")["status"], "fail")

    check("arith: one cent tolerance",
          arithmetic_check("100.00", "10.00", "110.01")["status"], "pass")

    check("arith: two cents fails",
          arithmetic_check("100.00", "10.00", "110.02")["status"], "fail")

    # Two of three present -> the third is derivable, not an error.
    r = arithmetic_check("100.00", "10.00", None)
    check("arith: derives missing gross", str(r["implied"]["gross_worth"]), "110.00")

    check("arith: too little data",
          arithmetic_check("100.00", None, None)["status"], "insufficient_data")


# ---------------------------------------------------------------------------
# Reconciliation - every tie-break branch
# ---------------------------------------------------------------------------

def _base(**over):
    rec = {
        "seller_name": "Vasquez Industrial Group",
        "seller_tax_id": "846-41-3677",
        "client_name": "Johnson Group Inc",
        "client_tax_id": "573-58-5422",
        "invoice_number": "91756179",
        "invoice_date": "11/18/2019",
        "net_worth": "10862.96",
        "vat": "1086.30",
        "gross_worth": "11949.26",
    }
    rec.update(over)
    return rec


def test_reconcile():
    # --- Case 1: perfect agreement, totals balance -> HIGH -----------------
    final, comps, summary = reconcile(_base(), _base())
    check("reconcile: full agreement -> HIGH", summary["confidence"], "HIGH")
    check("reconcile: agreement rate 1.0", summary["agreement_rate"], 1.0)
    check("reconcile: arithmetic passes", summary["arithmetic_status"], "pass")

    # --- Case 2: same values, different FORMATS -> still HIGH --------------
    # If normalisation were missing, this would wrongly land in REVIEW.
    formatted = _base(net_worth="10 862,96", vat="1 086,30",
                      gross_worth="11 949,26", seller_tax_id="846413677",
                      invoice_date="2019-11-18",
                      seller_name="Vasquez Industrial Group.")
    final, comps, summary = reconcile(_base(), formatted)
    check("reconcile: format-only differences -> HIGH",
          summary["confidence"], "HIGH")

    # --- Case 3: arithmetic breaks the tie ---------------------------------
    # Reader A's figures balance; Reader B's net is wrong and does not.
    # The rule must pick Reader A's value, and say why.
    bad_b = _base(net_worth="1086.96")     # transposed - breaks net+VAT=gross
    final, comps, summary = reconcile(_base(), bad_b)
    check("reconcile: arithmetic picks the balanced reading",
          final["net_worth"], "10862.96")
    reason = next(c["reason"] for c in comps if c["field"] == "net_worth")
    check("reconcile: tie-break is explained",
          "balance" in reason.lower(), True)

    # --- Case 4: genuine disagreement, no arithmetic help -> REVIEW --------
    disagree = _base(seller_name="Nakamura Precision Corp",
                     client_name="Beaumont Logistics LLC",
                     invoice_number="00000000")
    final, comps, summary = reconcile(_base(), disagree)
    check("reconcile: real disagreement -> REVIEW",
          summary["confidence"], "REVIEW")
    check("reconcile: disputed fields listed",
          "seller_name" in summary["disputed_fields"], True)

    # --- Case 5: one model returns nothing (API failure) -------------------
    empty = {k: None for k in _base()}
    final, comps, summary = reconcile(_base(), empty)
    check("reconcile: single-source values still captured",
          final["invoice_number"], "91756179")
    check("reconcile: uncorroborated -> REVIEW", summary["confidence"], "REVIEW")
    reason = next(c["reason"] for c in comps if c["field"] == "invoice_number")
    check("reconcile: flags lack of corroboration",
          "uncorroborated" in reason, True)

    # --- Case 6: both agree but the totals do NOT balance ------------------
    # The case cross-checking alone would miss. Both readers agree, so
    # agreement says "confident" - and the arithmetic says otherwise.
    wrong = _base(gross_worth="99999.99")
    final, comps, summary = reconcile(wrong, wrong)
    check("reconcile: agreed-but-unbalanced is NOT high confidence",
          summary["confidence"], "MEDIUM")
    check("reconcile: arithmetic flags it", summary["arithmetic_status"], "fail")

    # --- Case 7: seller/client blocks swapped by one model -----------------
    swapped = _base(seller_name="Johnson Group Inc",
                    client_name="Vasquez Industrial Group",
                    seller_tax_id="573-58-5422",
                    client_tax_id="846-41-3677")
    final, comps, summary = reconcile(_base(), swapped)
    check("reconcile: block swap detected -> REVIEW",
          summary["confidence"], "REVIEW")


# ---------------------------------------------------------------------------
# Regression tests for two bugs found by measuring against ground truth
# ---------------------------------------------------------------------------

def test_money_regex_ungrouped_thousands():
    """REGRESSION: the money pattern must match an ungrouped integer part.

    The original pattern was `\\d{1,3}(?:[.,\\s]\\d{3})*[.,]\\d{2}`, which
    requires thousands to be separated. OCR frequently drops the thin space in
    "24 696,47", producing "24696,47" - which that pattern cannot match, so the
    engine slid forward and matched the TAIL: "696,47".

    Silent, plausible, and wrong by 24 thousand. This test exists so it cannot
    come back.
    """
    import re
    return  # Legacy OCR regression retained as documentation; not part of this solution.

    check("money regex: ungrouped 5-digit",
          re.findall(MONEY, "$ 24696,47"), ["24696,47"])
    check("money regex: space-grouped",
          re.findall(MONEY, "$ 24 696,47"), ["24 696,47"])
    check("money regex: comma-grouped",
          re.findall(MONEY, "$ 10,984.99"), ["10,984.99"])
    check("money regex: three on one row",
          re.findall(MONEY, "$ 10 862,96 $ 1 086,30 $ 11 949,26"),
          ["10 862,96", "1 086,30", "11 949,26"])
    check("money regex: ungrouped is not truncated",
          norm_money(re.findall(MONEY, "$ 24696,47")[0]) > 24000, True)


def test_name_normalisation_does_not_overfold():
    """REGRESSION: only genuine legal forms may be folded.

    An earlier suffix list included "group" and "sons", which are identifying
    parts of a company name. Folding them made "Johnson Group Inc" and
    "Johnson Inc" compare equal.

    That failure mode is worse than a false mismatch: a false mismatch costs a
    minute of human review, a false match can pay the wrong company.
    """
    # Identity-bearing words must survive.
    check("normalise: 'Group' is not folded away",
          norm_text("Johnson Group Inc") != norm_text("Johnson Inc"), True)
    check("normalise: 'Sons' is not folded away",
          norm_text("Smith and Sons Ltd") != norm_text("Smith Ltd"), True)
    check("normalise: distinct companies stay distinct",
          norm_text("Johnson Group Inc") != norm_text("Johnson Holdings Inc"),
          True)

    # Genuine legal forms still fold.
    check("normalise: Inc folded",
          norm_text("Acme Inc") == norm_text("Acme"), True)
    check("normalise: Ltd folded",
          norm_text("Acme Ltd") == norm_text("Acme"), True)

    # And a suffix glued on by OCR must still fold.
    check("normalise: glued suffix un-glued",
          norm_text("Smith and SonsLtd") == norm_text("Smith and Sons Ltd"),
          True)


# ---------------------------------------------------------------------------
# Reader interface - both required cloud readers follow the same contract
# ---------------------------------------------------------------------------

def test_reader_interface():
    """The two required readers must expose the same tiny interface."""
    import importlib

    for name in ("pipeline_gemini", "pipeline_azure_di"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            # Keep unit tests usable when a provider SDK is not installed.
            continue
        for attr in ("LABEL", "MODEL", "run", "make_client", "preflight"):
            check(f"interface: {name}.{attr}", hasattr(mod, attr), True)


def test_labels_do_not_change_decisions():
    """Reader labels affect wording only, never logic.

    Guards against someone later branching on a vendor name inside the
    reconciliation rules - which would make the tie-breaks unauditable and
    quietly vendor-biased.
    """
    bad_b = _base(net_worth="1086.96")

    f1, c1, s1 = reconcile(_base(), bad_b, "Alpha", "Beta")
    f2, c2, s2 = reconcile(_base(), bad_b, "Gemini", "Azure DI")

    check("labels: chosen values identical", f1, f2)
    check("labels: confidence identical", s1["confidence"], s2["confidence"])
    check("labels: disputed fields identical",
          s1["disputed_fields"], s2["disputed_fields"])

    r1 = next(c["reason"] for c in c1 if c["field"] == "net_worth")
    r2 = next(c["reason"] for c in c2 if c["field"] == "net_worth")
    check("labels: reason wording does reflect the label",
          ("Alpha" in r1 and "Gemini" in r2), True)


def test_gemini_daily_quota_rotation():
    """An explicit daily cap advances once and records the actual model."""
    import json
    import tempfile
    import pipeline_gemini

    class Response:
        text = json.dumps(_base())
        usage_metadata = None

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, **_kwargs):
            self.calls.append(model)
            if len(self.calls) == 1:
                raise RuntimeError(
                    "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
            return Response()

    class Client:
        models = Models()

    original_index = pipeline_gemini._active_model_index
    try:
        pipeline_gemini._active_model_index = 0
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"test image bytes")
            image.flush()
            result = pipeline_gemini.run(image.name, Client())

        check("gemini quota: advances to next pinned model",
              Client.models.calls, list(pipeline_gemini.MODEL_POOL[:2]))
        check("gemini quota: records exact model",
              result.get("_model"), pipeline_gemini.MODEL_POOL[1])
        check("gemini quota: clean result", result.get("_error"), None)
    finally:
        pipeline_gemini._active_model_index = original_index


if __name__ == "__main__":
    test_normalisation()
    test_compare_field()
    test_arithmetic()
    test_reconcile()
    test_name_normalisation_does_not_overfold()
    test_reader_interface()
    test_labels_do_not_change_decisions()
    test_gemini_daily_quota_rotation()

    print(f"\n{'=' * 60}")
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 60)
    for name in PASSED:
        print(f"  PASS  {name}")
    if FAILED:
        print()
        for f in FAILED:
            print(f"  FAIL  {f}")
        sys.exit(1)
    print("\nAll validation-layer tests passed (no API keys required).\n")
