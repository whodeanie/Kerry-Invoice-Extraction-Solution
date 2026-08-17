"""
Orchestrator - runs two READERS over a folder of invoices and writes the two
required deliverables.

    python src/run.py --reader-a gemini --reader-b azure

Both readers expose the same small interface:

    LABEL              display name
    MODEL              pinned model identifier
    make_client()      build an authenticated client
    preflight(client)  one cheap call proving credentials work; raises on fail
    run(path, client)  image path in, 9 fields out; NEVER raises

The validation layer compares two records and does not depend on provider-
specific response shapes.

Operational notes:

* The two readers run CONCURRENTLY per image, and images run through a thread
  pool. Extraction is I/O-bound, so 50 images take about as long as the slowest
  few rather than the sum of all of them.

* Every raw response is cached to disk as JSON. Reruns of the analysis cost
  nothing because extraction is not repeated, and the cache IS the audit
  record - each reader's untouched original output, preserved before any
  reconciliation logic touched it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from fields import FIELDS
from validate import reconcile

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Readers are imported lazily so credential and SDK errors are reported during
# preflight rather than at module import time.
READERS = {
    "gemini": ("pipeline_gemini", None),
    "azure":  ("pipeline_azure_di", None),
}

# Azure's free tier (F0) permits 1 transaction/second. Exceeding it produces
# throttling that looks like extraction failure, so we cap parallelism when
# Azure is in play rather than letting the user discover this as a bug.
RATE_CAPPED = {"azure": 1}


def load_reader(name: str):
    if name not in READERS:
        sys.exit(f"ERROR: unknown reader '{name}'. "
                 f"Choose from: {', '.join(READERS)}")
    module_name, _ = READERS[name]
    try:
        return __import__(module_name)
    except ImportError as exc:
        sys.exit(f"ERROR: reader '{name}' needs a package that is not "
                 f"installed ({exc}). Try: pip install -r requirements.txt")


def find_images(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        sys.exit(f"ERROR: {folder} is not a directory")
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)


def process_one(path: Path, cache_dir: Path, reader_a, reader_b,
                client_a, client_b, key_a: str, key_b: str,
                use_cache: bool = True) -> dict:
    """Run both readers on one image and reconcile the two readings."""
    cache_file = cache_dir / f"{path.stem}__{key_a}_vs_{key_b}.json"

    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text())
        rec_a, rec_b = raw["reader_a"], raw["reader_b"]
    else:
        # Both readers see the image at the same time and neither sees the
        # other's answer. Independence is the whole point - if one could see
        # the other, agreement would prove nothing.
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(reader_a.run, str(path), client_a)
            fut_b = pool.submit(reader_b.run, str(path), client_b)
            rec_a, rec_b = fut_a.result(), fut_b.result()

        # Only cache a CLEAN result. Caching a failure would be worse than not
        # caching at all: a transient outage or an expired key would be frozen
        # into the cache, and every later rerun would silently replay the
        # failure instead of retrying it. Failures must stay retryable.
        if not rec_a.get("_error") and not rec_b.get("_error"):
            cache_file.write_text(json.dumps(
                {"image": path.name,
                 "reader_a": rec_a, "reader_b": rec_b,
                 "reader_a_name": key_a, "reader_b_name": key_b},
                indent=2, default=str))

    final, comparisons, summary = reconcile(
        rec_a, rec_b, reader_a.LABEL, reader_b.LABEL)

    return {
        "image": path.name,
        "final": final,
        "comparisons": comparisons,
        "summary": summary,
        "errors": {reader_a.LABEL: rec_a.get("_error"),
                   reader_b.LABEL: rec_b.get("_error")},
        "tokens": (rec_a.get("_tokens") or 0) + (rec_b.get("_tokens") or 0),
        # Azure supplies calibrated per-field confidence; the LLM readers do
        # not. Carried through so it can inform a disputed field.
        "confidence_a": rec_a.get("_confidence") or {},
        "confidence_b": rec_b.get("_confidence") or {},
    }


def write_outputs(results: list[dict], out_dir: Path,
                  label_a: str, label_b: str) -> None:
    """Write the two deliverables the brief asks for."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- output.csv : one row per invoice, the final answer ----------------
    with open(out_dir / "output.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_file"] + FIELDS
                   + ["confidence", "agreement_rate", "arithmetic_status",
                      "disputed_fields", "flags"])
        for r in results:
            w.writerow(
                [r["image"]]
                + [r["final"].get(f) if r["final"].get(f) is not None else ""
                   for f in FIELDS]
                + [r["summary"]["confidence"],
                   r["summary"]["agreement_rate"],
                   r["summary"]["arithmetic_status"],
                   r["summary"]["disputed_fields"],
                   r["summary"]["plausibility_flags"]]
            )

    # --- comparison_report.csv : one row per invoice PER FIELD -------------
    # Long format rather than wide, because it is what an auditor actually
    # wants: filter to status=mismatch and you have the complete exception
    # list, with both readings and the reason for the chosen value on each row.
    with open(out_dir / "comparison_report.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_file", "field",
                    f"{label_a}_value", f"{label_b}_value",
                    f"{label_a}_normalised", f"{label_b}_normalised",
                    "status", "similarity",
                    f"{label_b}_field_confidence",
                    "chosen_value", "reason"])
        for r in results:
            conf_b = r.get("confidence_b", {})
            for c in r["comparisons"]:
                w.writerow([
                    r["image"], c["field"],
                    c["value_a"] if c["value_a"] is not None else "",
                    c["value_b"] if c["value_b"] is not None else "",
                    c["normalised_a"], c["normalised_b"],
                    c["status"], c["similarity"],
                    conf_b.get(c["field"], ""),
                    c.get("chosen") if c.get("chosen") is not None else "",
                    c.get("reason", ""),
                ])


def print_summary(results: list[dict], elapsed: float,
                  label_a: str, label_b: str) -> None:
    """The numbers to read out in the meeting."""
    n = len(results)
    if not n:
        return
    tiers = {"HIGH": 0, "MEDIUM": 0, "REVIEW": 0}
    for r in results:
        tiers[r["summary"]["confidence"]] += 1

    field_match = {f: 0 for f in FIELDS}
    for r in results:
        for c in r["comparisons"]:
            if c["status"] == "match":
                field_match[c["field"]] += 1

    arith_pass = sum(1 for r in results
                     if r["summary"]["arithmetic_status"] == "pass")
    errors = sum(1 for r in results if any(r["errors"].values()))
    tok = sum(r["tokens"] for r in results)

    print("\n" + "=" * 64)
    print(f"  PROCESSED {n} INVOICES in {elapsed:.1f}s "
          f"({elapsed / n:.1f}s per invoice)")
    print("=" * 64)
    print("\n  CONFIDENCE TIERS")
    for tier, count in tiers.items():
        pct = 100 * count / n
        bar = "#" * int(pct / 2.5)
        note = {"HIGH": "high confidence", "MEDIUM": "spot-check",
                "REVIEW": "human review"}[tier]
        print(f"    {tier:<7} {count:>3} ({pct:5.1f}%)  {bar:<40} {note}")

    print(f"\n  PER-FIELD AGREEMENT  ({label_a} vs {label_b})")
    for f in FIELDS:
        pct = 100 * field_match[f] / n
        bar = "#" * int(pct / 2.5)
        print(f"    {f:<18} {pct:5.1f}%  {bar}")

    print(f"\n  ARITHMETIC TIE-OUT   {arith_pass}/{n} "
          f"({100 * arith_pass / n:.1f}%) net+VAT=gross")
    print(f"  EXTRACTION ERRORS    {errors}")
    if tok:
        print(f"  TOTAL TOKENS         {tok:,}")

    # Surface distinct error messages. A run reporting "0/9 fields agree"
    # across the board is almost never a model problem - it is usually
    # credentials or billing, and the operator needs to see that immediately
    # rather than infer it from a wall of zeroes.
    if errors:
        distinct = {}
        for r in results:
            for reader, err in r["errors"].items():
                if err:
                    distinct.setdefault(str(err)[:160], set()).add(reader)
        print("\n  ERRORS ENCOUNTERED")
        for msg, readers in distinct.items():
            print(f"    [{'/'.join(sorted(readers))}] {msg}")
    print("=" * 64 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dual-reader invoice extraction with cross-validation")
    ap.add_argument("--reader-a", default="gemini", choices=list(READERS),
                    help="first independent reader (default: gemini)")
    ap.add_argument("--reader-b", default="azure", choices=list(READERS),
                    help="second independent reader (default: azure)")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache", default="output/raw_cache")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.reader_a == args.reader_b:
        sys.exit("ERROR: the two readers must be different - that is the "
                 "entire point. Two copies of the same reader make the same "
                 "mistakes, so their agreement proves nothing.")

    load_dotenv()

    reader_a, reader_b = load_reader(args.reader_a), load_reader(args.reader_b)

    images = find_images(args.images)
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"ERROR: no images found in {args.images}")

    # Respect the slowest reader's rate limit rather than letting throttling
    # masquerade as extraction failure.
    workers = args.workers
    for name in (args.reader_a, args.reader_b):
        if name in RATE_CAPPED:
            cap = RATE_CAPPED[name]
            if workers > cap:
                print(f"NOTE: '{name}' is rate-limited on its free tier - "
                      f"reducing workers from {workers} to {cap}.\n")
                workers = cap

    # Build clients + preflight. One tiny call each before committing to a
    # batch: a bad key or empty quota otherwise produces 50 identical failures
    # and a confusing report. Fail fast on two cheap calls, not slowly on 100.
    clients, errors = {}, []
    for tag, reader in (("A", reader_a), ("B", reader_b)):
        try:
            clients[tag] = reader.make_client()
            reader.preflight(clients[tag])
        except KeyError as exc:
            errors.append(f"{reader.LABEL}: environment variable {exc} not set")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reader.LABEL}: {exc}")

    if errors:
        print("PREFLIGHT FAILED - not starting the batch:\n")
        for e in errors:
            print(f"  * {e}\n")
        print("The runtime environment is not configured for this provider.")
        sys.exit(1)

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} invoices")
    print(f"  Reader A: {reader_a.LABEL} ({reader_a.MODEL})")
    print(f"  Reader B: {reader_b.LABEL} ({reader_b.MODEL})")
    print(f"  Workers:  {workers}\n")

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one, p, cache_dir, reader_a, reader_b,
                        clients["A"], clients["B"],
                        args.reader_a, args.reader_b, not args.no_cache): p
            for p in images
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            mark = {"HIGH": "+", "MEDIUM": "~", "REVIEW": "!"}[
                res["summary"]["confidence"]]
            print(f"  [{i:>3}/{len(images)}] {mark} {res['image']:<28} "
                  f"{res['summary']['confidence']:<7} "
                  f"{res['summary']['fields_matched']}/9 fields agree")

    elapsed = time.time() - start
    results.sort(key=lambda r: r["image"])

    write_outputs(results, Path(args.out), reader_a.LABEL, reader_b.LABEL)
    print_summary(results, elapsed, reader_a.LABEL, reader_b.LABEL)
    print(f"Wrote {args.out}/output.csv and {args.out}/comparison_report.csv")


if __name__ == "__main__":
    main()
