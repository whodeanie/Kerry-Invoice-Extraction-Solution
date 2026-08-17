"""Visual dashboard for the dual-reader invoice extraction pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fields import FIELDS  # noqa: E402
from run import RATE_CAPPED, process_one, write_outputs  # noqa: E402
READER_MODULES = {
    "gemini": "pipeline_gemini",
    "azure": "pipeline_azure_di",
}

st.set_page_config(
    page_title="Invoice Cross-Check",
    page_icon="IX",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #14213d;
        --muted: #64748b;
        --line: #dfe7f0;
        --paper: #ffffff;
        --canvas: #f3f6fa;
        --teal: #0f766e;
        --teal-soft: #ccfbf1;
        --amber: #b45309;
        --amber-soft: #fef3c7;
        --red: #be123c;
        --red-soft: #ffe4e6;
      }
      .stApp {background: var(--canvas); color: var(--ink);}
      .block-container {max-width: 1360px; padding: 1.4rem 2.2rem 4rem;}
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c172a 0%, #14213d 100%);
        border-right: 0;
      }
      [data-testid="stSidebar"] * {color: #e8eef7;}
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-baseweb="input"] > div,
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.08);
        border-color: rgba(255,255,255,.16);
      }
      [data-testid="stSidebar"] .stButton button {
        min-height: 3rem; border: 0; border-radius: 12px;
        background: #2dd4bf; color: #082f2a; font-weight: 800;
        box-shadow: 0 10px 24px rgba(45,212,191,.18);
      }
      [data-testid="stSidebar"] .stButton button:hover {background: #5eead4;}
      header[data-testid="stHeader"] {background: transparent;}
      #MainMenu, footer {visibility: hidden;}
      h1, h2, h3 {color: var(--ink); letter-spacing: -.035em;}
      .brand-row {display:flex; align-items:center; justify-content:space-between;
        gap:1rem; padding:.2rem 0 1.2rem; border-bottom:1px solid var(--line);}
      .brand-lockup {display:flex; align-items:center; gap:.8rem;}
      .brand-mark {display:grid; place-items:center; width:38px; height:38px;
        background:var(--ink); color:white; border-radius:11px; font-size:.75rem;
        font-weight:900; letter-spacing:.06em; box-shadow:0 6px 14px rgba(20,33,61,.18);}
      .brand-name {font-weight:850; font-size:1.02rem; color:var(--ink);}
      .brand-note {font-size:.72rem; color:var(--muted); margin-top:.08rem;}
      .live-pill {display:inline-flex; align-items:center; gap:.45rem; padding:.45rem .72rem;
        border:1px solid #a7f3d0; background:#ecfdf5; color:#047857;
        border-radius:999px; font-size:.73rem; font-weight:750;}
      .live-dot {width:7px; height:7px; border-radius:50%; background:#10b981;
        box-shadow:0 0 0 4px rgba(16,185,129,.12);}
      .hero {padding:2.2rem 0 1.5rem; display:grid; grid-template-columns:1.3fr .7fr;
        gap:2rem; align-items:end;}
      .eyebrow {color:var(--teal); font-size:.72rem; font-weight:850;
        letter-spacing:.14em; text-transform:uppercase; margin-bottom:.7rem;}
      .hero h1 {font-size:clamp(2.1rem,3.6vw,3.25rem); line-height:1.02; margin:0;
        max-width:850px;}
      .hero-copy {color:var(--muted); font-size:1rem; line-height:1.65;
        max-width:480px; margin:0 0 .2rem auto;}
      .signal-strip {display:grid; grid-template-columns:repeat(2,1fr); gap:.8rem;
        margin:0 0 1.35rem;}
      .signal {background:rgba(255,255,255,.78); backdrop-filter:blur(10px);
        border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem;
        display:flex; align-items:center; justify-content:space-between; gap:.6rem;}
      .signal-name {font-size:.78rem; font-weight:800; color:var(--ink);}
      .signal-detail {font-size:.7rem; color:var(--muted); margin-top:.12rem;}
      .signal-state {font-size:.65rem; color:var(--teal); background:var(--teal-soft);
        border-radius:999px; padding:.3rem .48rem; font-weight:850; text-transform:uppercase;}
      .section-kicker {font-size:.7rem; color:var(--teal); text-transform:uppercase;
        letter-spacing:.12em; font-weight:850; margin-bottom:.35rem;}
      .section-title {font-size:1.45rem; font-weight:850; color:var(--ink); margin-bottom:.15rem;}
      .section-copy {font-size:.82rem; color:var(--muted); margin-bottom:1rem;}
      .metric-grid {display:grid; grid-template-columns:repeat(5,1fr); gap:.75rem; margin:1rem 0;}
      .metric-card {background:var(--paper); border:1px solid var(--line); border-radius:16px;
        padding:1rem 1.05rem; box-shadow:0 6px 20px rgba(20,33,61,.04);}
      .metric-label {font-size:.68rem; color:var(--muted); font-weight:750;
        text-transform:uppercase; letter-spacing:.08em;}
      .metric-value {font-size:1.7rem; color:var(--ink); font-weight:900;
        letter-spacing:-.05em; margin-top:.25rem;}
      .metric-sub {font-size:.68rem; color:var(--muted); margin-top:.15rem;}
      .routing-card {background:linear-gradient(135deg,#14213d 0%,#1e3a5f 100%);
        color:white; border-radius:18px; padding:1.1rem 1.2rem; margin:.8rem 0 1rem;
        display:grid; grid-template-columns:auto 1fr; gap:1.2rem; align-items:center;
        box-shadow:0 12px 32px rgba(20,33,61,.14);}
      .routing-score {font-size:2.1rem; line-height:1; font-weight:900; color:#5eead4;}
      .routing-label {font-size:.68rem; color:#a5b4cf; text-transform:uppercase;
        letter-spacing:.08em; margin-top:.2rem;}
      .route-bar {display:flex; height:8px; overflow:hidden; border-radius:999px;
        background:rgba(255,255,255,.12);}
      .route-high {background:#2dd4bf;}.route-medium {background:#fbbf24;}.route-review {background:#fb7185;}
      .route-legend {display:flex; gap:1.1rem; margin-top:.55rem; font-size:.68rem; color:#cbd5e1;}
      .legend-dot {display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.35rem;}
      .invoice-meta {display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 1rem;}
      .meta-pill {border:1px solid var(--line); background:white; color:var(--ink);
        border-radius:999px; padding:.38rem .62rem; font-size:.7rem; font-weight:750;}
      .meta-pill.good {border-color:#99f6e4; background:#f0fdfa; color:#0f766e;}
      .sidebar-brand {padding:.15rem 0 .8rem; font-size:1.05rem; font-weight:900;
        letter-spacing:-.02em;}
      .sidebar-step {font-size:.66rem; color:#94a3b8 !important; text-transform:uppercase;
        letter-spacing:.12em; font-weight:800; margin:1rem 0 .4rem;}
      .dataset-card {border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.07);
        border-radius:14px; padding:.85rem .9rem;}
      .dataset-count {font-size:1.55rem; line-height:1; font-weight:900; color:#5eead4;}
      .dataset-title {font-size:.78rem; font-weight:800; color:#f8fafc; margin-top:.35rem;}
      .dataset-range {font-size:.66rem; color:#94a3b8; margin-top:.2rem;}
      .empty-card {background:white; border:1px solid var(--line); border-radius:20px;
        padding:2rem; box-shadow:0 8px 30px rgba(20,33,61,.05);}
      .empty-number {font-size:3.2rem; font-weight:900; color:#dbe5ef; line-height:1;}
      div[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:14px; overflow:hidden;}
      div[data-baseweb="tab-list"] {gap:.35rem; background:#e9eef5; padding:.28rem;
        border-radius:12px; width:max-content;}
      button[data-baseweb="tab"] {border-radius:9px; padding:.55rem 1rem;}
      button[data-baseweb="tab"][aria-selected="true"] {background:white; box-shadow:0 2px 8px rgba(20,33,61,.08);}
      @media (max-width: 1200px) {
        .hero {grid-template-columns:1fr; gap:.8rem}.hero-copy {margin:0}
        .signal-strip {grid-template-columns:1fr}
      }
      @media (max-width: 900px) {
        .metric-grid {grid-template-columns:1fr 1fr}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def reader(name: str):
    return importlib.import_module(READER_MODULES[name])


def preflight_pair(name_a: str, name_b: str):
    # Streamlit is a long-running process. Reload with override so credentials
    # corrected on disk take effect without restarting the dashboard server.
    load_dotenv(ROOT / ".env", override=True)
    loaded = [reader(name_a), reader(name_b)]
    clients = []
    failures = []
    for module in loaded:
        try:
            client = module.make_client()
            module.preflight(client)
            clients.append(client)
        except Exception as exc:  # noqa: BLE001
            clients.append(None)
            failures.append(f"{module.LABEL}: {exc}")
    if failures:
        raise RuntimeError("\n\n".join(failures))
    return loaded[0], loaded[1], clients[0], clients[1]


def run_pair(paths: list[Path], name_a: str, name_b: str) -> tuple[list[dict], float]:
    module_a, module_b, client_a, client_b = preflight_pair(name_a, name_b)
    results = []
    started = time.time()
    cache_dir = ROOT / "output" / "raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0, text="Readers passed preflight…")
    for index, path in enumerate(paths, 1):
        progress.progress(
            index / len(paths), text=f"Cross-checking {path.name} ({index}/{len(paths)})"
        )
        results.append(process_one(
            path, cache_dir, module_a, module_b, client_a, client_b,
            name_a, name_b, True,
        ))
        if name_a in RATE_CAPPED or name_b in RATE_CAPPED:
            time.sleep(1.05)
    progress.empty()
    return results, time.time() - started


def run_single(paths: list[Path], name: str) -> tuple[list[dict], float]:
    """Run one extraction approach independently."""
    load_dotenv(ROOT / ".env", override=True)
    module = reader(name)
    client = module.make_client()
    module.preflight(client)
    cache_dir = ROOT / "output" / "raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    progress = st.progress(0, text=f"{module.LABEL} passed preflight…")
    for index, path in enumerate(paths, 1):
        progress.progress(
            index / len(paths),
            text=f"Running {module.LABEL}: {path.name} ({index}/{len(paths)})",
        )
        cache_file = cache_dir / f"{path.stem}__{name}.json"
        if cache_file.exists():
            record = json.loads(cache_file.read_text())
        else:
            record = module.run(str(path), client)
            if not record.get("_error"):
                cache_file.write_text(json.dumps(record, indent=2, default=str))
        results.append({
            "image": path.name,
            "record": record,
            "error": record.get("_error"),
            "confidence": record.get("_confidence") or {},
        })
        if name in RATE_CAPPED:
            time.sleep(1.05)
    progress.empty()
    return results, time.time() - started


def provider_configured(name: str) -> bool:
    requirements = {
        "gemini": ("GEMINI_API_KEY",),
        "azure": ("AZURE_DI_ENDPOINT", "AZURE_DI_KEY"),
    }
    for key in requirements[name]:
        value = os.environ.get(key, "").strip()
        if not value or any(marker in value.lower() for marker in
                            ("resource-name", "your-", "...", "<", ">")):
            return False
    return True


def download_buttons() -> None:
    output_csv = ROOT / "output" / "output.csv"
    comparison_csv = ROOT / "output" / "comparison_report.csv"
    if not output_csv.exists() or not comparison_csv.exists():
        return
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download verified records", output_csv.read_bytes(),
        "verified-invoices.csv", "text/csv", width="stretch",
    )
    downloads[1].download_button(
        "Download audit trail", comparison_csv.read_bytes(),
        "invoice-audit-trail.csv", "text/csv", width="stretch",
    )


def show_single_results(results: list[dict], elapsed: float, name: str) -> None:
    module = reader(name)
    total_fields = len(results) * len(FIELDS)
    found = sum(
        value not in (None, "")
        for result in results
        for field, value in result["record"].items()
        if field in FIELDS
    )
    errors = sum(bool(result["error"]) for result in results)
    st.markdown('<div class="section-kicker">Single approach</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{module.LABEL} extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">This view shows one extraction approach in isolation. '
        'Confidence routing requires the Cross-check both mode.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Invoices</div><div class="metric-value">{len(results)}</div></div>
          <div class="metric-card"><div class="metric-label">Fields found</div><div class="metric-value">{found}/{total_fields}</div></div>
          <div class="metric-card"><div class="metric-label">Provider errors</div><div class="metric-value">{errors}</div></div>
          <div class="metric-card"><div class="metric-label">Run time</div><div class="metric-value">{elapsed:.1f}s</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
    rows = []
    for result in results:
        row = {"Invoice": result["image"]}
        row.update({field.replace("_", " ").title(): result["record"].get(field) for field in FIELDS})
        row["Error"] = result["error"] or ""
        rows.append(row)
    overview_tab, detail_tab = st.tabs(["Extraction results", "Invoice detail"])
    with overview_tab:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with detail_tab:
        selected = st.selectbox(
            "Invoice to inspect", [result["image"] for result in results],
            key=f"single_invoice_{name}",
        )
        result = next(item for item in results if item["image"] == selected)
        left, right = st.columns([0.78, 1.42], gap="large")
        with left:
            st.image(str(ROOT / "data" / "images" / selected), width="stretch")
        with right:
            detail_rows = []
            for field in FIELDS:
                detail_rows.append({
                    "Field": field.replace("_", " ").title(),
                    "Extracted value": result["record"].get(field),
                    "Provider confidence": result["confidence"].get(field, "—"),
                })
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")


def show_results(results: list[dict], elapsed: float,
                 labels: tuple[str, str] = ("Reader A", "Reader B")) -> None:
    tiers = {tier: sum(r["summary"]["confidence"] == tier for r in results)
             for tier in ("HIGH", "MEDIUM", "REVIEW")}
    agreement = sum(r["summary"]["agreement_rate"] for r in results) / len(results)
    arithmetic = sum(r["summary"]["arithmetic_status"] == "pass" for r in results)
    disputed = sum(bool(r["summary"]["disputed_fields"]) for r in results)
    total = len(results)
    high_width = 100 * tiers["HIGH"] / total
    medium_width = 100 * tiers["MEDIUM"] / total
    review_width = 100 * tiers["REVIEW"] / total

    st.markdown('<div class="section-kicker">Completed batch</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Verification overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-copy">{labels[0]} cross-checked against {labels[1]}. '
        'Every selected value retains its decision reason.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Invoices</div>
            <div class="metric-value">{total}</div><div class="metric-sub">processed in this batch</div></div>
          <div class="metric-card"><div class="metric-label">Field agreement</div>
            <div class="metric-value">{agreement:.0%}</div><div class="metric-sub">across nine required fields</div></div>
          <div class="metric-card"><div class="metric-label">Arithmetic</div>
            <div class="metric-value">{arithmetic}/{total}</div><div class="metric-sub">net + VAT = gross</div></div>
          <div class="metric-card"><div class="metric-label">Disputed</div>
            <div class="metric-value">{disputed}</div><div class="metric-sub">invoices with exceptions</div></div>
          <div class="metric-card"><div class="metric-label">Run time</div>
            <div class="metric-value">{elapsed:.1f}s</div><div class="metric-sub">including provider checks</div></div>
        </div>
        <div class="routing-card">
          <div><div class="routing-score">{tiers['REVIEW']}</div>
            <div class="routing-label">manual reviews</div></div>
          <div><div class="route-bar">
            <span class="route-high" style="width:{high_width}%"></span>
            <span class="route-medium" style="width:{medium_width}%"></span>
            <span class="route-review" style="width:{review_width}%"></span>
          </div><div class="route-legend">
            <span><i class="legend-dot" style="background:#2dd4bf"></i>{tiers['HIGH']} high confidence</span>
            <span><i class="legend-dot" style="background:#fbbf24"></i>{tiers['MEDIUM']} spot-check</span>
            <span><i class="legend-dot" style="background:#fb7185"></i>{tiers['REVIEW']} review</span>
          </div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, exceptions_tab, audit_tab = st.tabs(
        ["Batch overview", "Exception queue", "Invoice audit"]
    )
    summary_rows = []
    for result in results:
        summary_rows.append({
            "Invoice": result["image"],
            "Route": result["summary"]["confidence"],
            "Agreement": result["summary"]["agreement_rate"],
            "Arithmetic": result["summary"]["arithmetic_status"].upper(),
            "Disputed fields": result["summary"]["disputed_fields"].replace(";", ", ") or "None",
        })

    with overview_tab:
        st.dataframe(
            pd.DataFrame(summary_rows), hide_index=True, width="stretch",
            column_config={
                "Agreement": st.column_config.ProgressColumn(
                    "Agreement", min_value=0.0, max_value=1.0, format="percent"
                ),
                "Route": st.column_config.TextColumn("Route", width="small"),
            },
        )
        download_buttons()

    with exceptions_tab:
        exceptions = []
        for result in results:
            for comparison in result["comparisons"]:
                if comparison["status"] != "match":
                    exceptions.append({
                        "Invoice": result["image"],
                        "Field": comparison["field"].replace("_", " ").title(),
                        labels[0]: comparison["value_a"] or "Not found",
                        labels[1]: comparison["value_b"] or "Not found",
                        "Decision": comparison.get("chosen") or "—",
                        "Why": comparison.get("reason") or "—",
                    })
        if exceptions:
            st.caption(f"{len(exceptions)} field exceptions across {disputed} invoices")
            st.dataframe(pd.DataFrame(exceptions), hide_index=True, width="stretch")
        else:
            st.success("No exceptions in this batch. Every field matched.")

    with audit_tab:
        selected = st.selectbox(
            "Invoice to inspect", [r["image"] for r in results],
            key="audit_invoice_selector",
        )
        result = next(r for r in results if r["image"] == selected)
        image_path = ROOT / "data" / "images" / selected
        summary = result["summary"]
        st.markdown(
            f'<div class="invoice-meta"><span class="meta-pill good">{summary["confidence"]} route</span>'
            f'<span class="meta-pill">{summary["fields_matched"]}/9 fields agree</span>'
            f'<span class="meta-pill">Arithmetic {summary["arithmetic_status"]}</span></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([0.78, 1.42], gap="large")
        with left:
            if image_path.exists():
                st.image(str(image_path), width="stretch")
        with right:
            comparison = pd.DataFrame(result["comparisons"]).rename(columns={
                "field": "Field", "value_a": labels[0], "value_b": labels[1],
                "status": "Status", "chosen": "Verified value", "reason": "Decision reason",
            })
            comparison["Field"] = comparison["Field"].str.replace("_", " ").str.title()
            columns = ["Field", labels[0], labels[1], "Status", "Verified value", "Decision reason"]
            st.dataframe(comparison[columns], hide_index=True, width="stretch")


def show_previous_run() -> None:
    output_csv = ROOT / "output" / "output.csv"
    if not output_csv.exists():
        st.markdown(
            '<div class="empty-card"><div class="empty-number">01</div>'
            '<h3>Ready for the first verification run</h3>'
            '<p>The assignment dataset is ready. Gemini and Azure will '
            'cross-check every field and route only genuine exceptions.</p></div>',
            unsafe_allow_html=True,
        )
        return
    previous = pd.read_csv(output_csv)
    high = int((previous["confidence"] == "HIGH").sum())
    review = int((previous["confidence"] == "REVIEW").sum())
    agreement = float(previous["agreement_rate"].mean())
    st.markdown('<div class="section-kicker">Most recent export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Last verified batch</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Invoices</div><div class="metric-value">{len(previous)}</div></div>
          <div class="metric-card"><div class="metric-label">High confidence</div><div class="metric-value">{high}</div></div>
          <div class="metric-card"><div class="metric-label">Needs review</div><div class="metric-value">{review}</div></div>
          <div class="metric-card"><div class="metric-label">Agreement</div><div class="metric-value">{agreement:.0%}</div></div>
          <div class="metric-card"><div class="metric-label">Arithmetic</div><div class="metric-value">{(previous['arithmetic_status'] == 'pass').sum()}/{len(previous)}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
    columns = ["image_file", "confidence", "agreement_rate", "arithmetic_status", "disputed_fields"]
    st.dataframe(previous[columns], hide_index=True, width="stretch")
    download_buttons()


load_dotenv(ROOT / ".env", override=True)

gemini_state = "Configured" if provider_configured("gemini") else "Needs key"
azure_state = "Configured" if provider_configured("azure") else "Needs setup"

st.markdown(
    f"""
    <div class="brand-row">
      <div class="brand-lockup"><div class="brand-mark">IX</div><div>
        <div class="brand-name">Invoice Cross-Check</div>
        <div class="brand-note">Auditable document intelligence</div>
      </div></div>
      <div class="live-pill"><span class="live-dot"></span>Pipeline online</div>
    </div>
    <div class="hero">
      <div><div class="eyebrow">Verification workspace</div>
        <h1>From invoice image to verified record.</h1></div>
      <p class="hero-copy">Two independent readers extract every field. A deterministic
        arithmetic check settles the numbers. You only review what truly disagrees.</p>
    </div>
    <div class="signal-strip">
      <div class="signal"><div><div class="signal-name">Gemini Flash</div>
        <div class="signal-detail">Generative vision reader</div></div><span class="signal-state">{gemini_state}</span></div>
      <div class="signal"><div><div class="signal-name">Azure Document Intelligence</div>
        <div class="signal-detail">Structured invoice extractor</div></div><span class="signal-state">{azure_state}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

invoice_paths = sorted(
    path for path in (ROOT / "data" / "images").iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)
invoice_names = [path.name for path in invoice_paths]


with st.sidebar:
    st.markdown('<div class="sidebar-brand">New verification run</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-step">01 · Invoice set</div>', unsafe_allow_html=True)
    invoice_count = len(invoice_names)
    st.markdown(
        f'<div class="dataset-card"><div class="dataset-count">{invoice_count} invoices</div>'
        '<div class="dataset-title">Complete assignment dataset</div>'
        '<div class="dataset-range">Invoices 0331–0380 · complete 50-file set</div></div>',
        unsafe_allow_html=True,
    )
    quick_run = st.toggle("Single-invoice preview", value=True)
    st.caption("On: 1 invoice · Off: all 50 invoices")
    st.markdown('<div class="sidebar-step">02 · Execution mode</div>', unsafe_allow_html=True)
    execution_mode = st.radio(
        "Approach",
        ["Cross-check both", "Gemini vision", "Azure structured"],
        captions=[
            "Run both independently, then validate",
            "Run only the vision-language approach",
            "Run only the purpose-built extractor",
        ],
        label_visibility="collapsed",
    )
    ready_a = "Configured" if provider_configured("gemini") else "Setup required"
    ready_b = "Configured" if provider_configured("azure") else "Setup required"
    st.caption(f"Gemini · {ready_a}   |   Azure · {ready_b}")
    st.markdown('<div class="sidebar-step">03 · Verify</div>', unsafe_allow_html=True)
    run_label = {
        "Cross-check both": "Run cross-check",
        "Gemini vision": "Run Gemini",
        "Azure structured": "Run Azure",
    }[execution_mode]
    run_clicked = st.button(run_label, type="primary", width="stretch")

if run_clicked:
    if not invoice_names:
        st.warning("No assignment invoices were found in data/images.")
    else:
        paths = invoice_paths[:1] if quick_run else invoice_paths
        try:
            with st.status("Running extraction…", expanded=True) as status:
                if execution_mode == "Cross-check both":
                    reader_a, reader_b = "gemini", "azure"
                    results, elapsed = run_pair(paths, reader_a, reader_b)
                    labels = (reader(reader_a).LABEL, reader(reader_b).LABEL)
                    # A presentation preview must never replace the validated
                    # full-batch deliverables with a one-row export.
                    if not quick_run:
                        write_outputs(results, ROOT / "output", *labels)
                    st.session_state["results"] = results
                    st.session_state["elapsed"] = elapsed
                    st.session_state["reader_labels"] = labels
                    st.session_state.pop("single_results", None)
                    st.session_state.pop("single_reader", None)
                    status.update(label="Cross-check complete", state="complete")
                else:
                    selected_reader = "gemini" if execution_mode == "Gemini vision" else "azure"
                    single_results, elapsed = run_single(paths, selected_reader)
                    st.session_state["single_results"] = single_results
                    st.session_state["single_reader"] = selected_reader
                    st.session_state["elapsed"] = elapsed
                    st.session_state.pop("results", None)
                    status.update(label=f"{reader(selected_reader).LABEL} extraction complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

if st.session_state.get("single_results"):
    show_single_results(
        st.session_state["single_results"], st.session_state["elapsed"],
        st.session_state["single_reader"],
    )
elif st.session_state.get("results"):
    show_results(
        st.session_state["results"], st.session_state["elapsed"],
        st.session_state.get("reader_labels", ("Reader A", "Reader B")),
    )
else:
    show_previous_run()
