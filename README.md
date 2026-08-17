# Kerry Invoice Extraction Solution

Extracts nine invoice fields with Gemini Flash and Azure Document Intelligence, then compares the results and validates the financial totals.

## Run

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The runtime must provide `GEMINI_API_KEY`, `AZURE_DI_ENDPOINT`, and `AZURE_DI_KEY`.

The Gemini adapter uses a fixed ordered list of Flash model versions when a
free-tier model reaches its daily request cap. The exact version used for each
invoice is retained in the raw audit response.

## Execution modes

- **Gemini vision** — runs the vision-language extraction approach.
- **Azure structured** — runs the prebuilt invoice extraction approach.
- **Cross-check both** — runs both independently, compares their fields, validates totals, and assigns a review tier.

`Single-invoice preview` processes the first invoice. Turn it off to process all 50 invoices.

## Data

`data/images/` contains the 50 assignment invoices, numbered `0331` through `0380`.

The extracted fields are seller name, seller tax ID, client name, client tax ID, invoice number, invoice date, net worth, VAT, and gross worth.

The images are a 50-file subset of the Kaggle dataset
[High-Quality Invoice Images for OCR](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr),
published under the Open Database License with Database Contents licensing.

## Validated assessment run

The included output was generated from all 50 assignment invoices: 24 HIGH,
26 MEDIUM, 0 REVIEW, 50/50 arithmetic tie-outs, and 0 provider errors. This is
agreement-based validation; it is not a claim of ground-truth accuracy.

## Output

Cross-check mode creates:

- `output/output.csv` — one reconciled row per invoice.
- `output/comparison_report.csv` — field-level readings, comparison status, selected value, and decision reason.
- `output/raw_cache/*.json` — successful provider responses used for reproducible reruns.

## Tests

The validation tests do not call either provider:

```bash
python tests/test_validation.py
```

## License

The source code is available under the MIT License. The invoice images retain
the licensing and attribution of the source Kaggle dataset described above.
