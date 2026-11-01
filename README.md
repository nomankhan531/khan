# khan# 📄 CV Analyzer & Job Matching System

A lightweight, privacy-first CV analysis tool built with **Python** and **Streamlit**.
It reads a CV (PDF, DOCX or TXT), extracts what can be reliably detected, and presents it in a clean dashboard.

> **Current status: Stage 1 – Foundation.** Job description analysis, match percentages,
> recommendations and PDF reports are planned for later stages (see [Roadmap](#roadmap)).

## Honesty principles

- **Detected facts** are matched directly from the CV text. **Estimates** are heuristics and can be wrong. The UI labels both.
- Nothing is invented: if something is not found, the app says "Not detected".
- Keyword matching is **not** a measure of human ability. A skill appearing in a CV is evidence of text, not of competence.

## Features (Stage 1)

| Area | What it does |
|------|--------------|
| Upload safety | Size limit (5 MB), extension + file-signature check, DOCX zip-bomb guard, PDF page limit |
| Extraction | PDF (`pypdf`), DOCX (`python-docx`, tables included in order), TXT (multi-encoding) |
| Contact | Emails, phones, LinkedIn, GitHub, other links (regex) |
| Structure | Recognises common section headings (Experience, Education, Skills, Projects, ...) |
| Skills | ~150 catalog skills with aliases; records *where* each appears |
| Experience | Date-range parsing, overlap-safe total (estimate) |
| Education | Lines containing degree keywords |
| Dashboard | Metrics, checklist, observations, chart, evidence table, JSON export |

## Installation

Requires **Python 3.10+**. Total install is small (suitable for 8 GB RAM laptops).

```bash
# 1. Get the project (after you push it to GitHub)
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate

# 3. Install dependencies
pip install -r REQUIREMENTS.TXT
```

## Usage

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501), upload a CV in the sidebar and explore the tabs.

### Run the built-in tests

```bash
python app.py --selftest
```

Every line should show `PASS`. Run this after any change to the code.

## Architecture

`app.py` is organised into numbered layers; each layer only depends on the ones above it:

```
1 Configuration      limits and constants
2 Data models        dataclasses: CVProfile, SkillHit, ContactInfo, ...
3 Errors             CVAnalyzerError -> FileValidationError / TextExtractionError
4 Validation+extract validate_upload -> extract_text (pdf | docx | txt)
5 Normalisation      normalize_text
6 Contact            regex extraction
7 Sections           heading lookup -> split_into_sections
8 Skills             alias catalog -> one compiled regex -> per-section counts
9 Experience/Edu     date ranges -> merged intervals -> months
10 Observations      plain-language parser findings
11 Pipeline          analyze_cv() ties everything together
12 UI                Streamlit rendering (no analysis logic inside)
13 Self-tests        run_self_tests()
```

**Key design rule:** the analysis functions (sections 4–11) never import or call Streamlit UI code, so they are testable without a browser.

## Privacy

- Files are processed **in memory**; nothing is written to disk.
- No network calls and no external services.
- Results live only in the Streamlit session and are removed by **Clear CV from memory** or by closing the session.
- `st.cache_data` is deliberately **not** used, because it could keep CV contents cached across sessions.
- CV text is shown with `st.text()` / HTML-escaped so it cannot inject markup.
- Do not deploy the app publicly with real candidate data unless you have a lawful basis and appropriate safeguards (for example consent, GDPR where applicable).

## Known limitations (Stage 1)

- Scanned/image-only PDFs have no text and are rejected (OCR is excluded to stay lightweight).
- Multi-column PDFs can scramble reading order.
- Section detection needs recognisable headings; unusual titles are not matched.
- Single-letter skills (C, R, Go) are intentionally not detected yet.
- Name detection is a heuristic estimate; the app may return "Not detected".
- Experience length ignores gaps, part-time work and dates written in other formats.

## Deployment (GitHub + Streamlit Community Cloud)

1. Create a new **public or private** repository on GitHub.
2. Add the three files at the repository root: `app.py`, the requirements file, and `README.md`.
3. Commit and push:
   ```bash
   git init
   git add app.py REQUIREMENTS.TXT README.md
   git commit -m "Stage 1: CV Analyzer foundation"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
4. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository, branch `main`, and main file `app.py`, then **Deploy**.

> ⚠️ **File-name case matters on Linux.** Streamlit Community Cloud runs on Linux and looks for a file named
> `requirements.txt` (lower-case). If your file is named `REQUIREMENTS.TXT`, dependencies may **not** be installed
> on the cloud. Before deploying, rename it: `git mv REQUIREMENTS.TXT requirements.txt`.

## Roadmap

| Stage | Goal |
|-------|------|
| 1 ✅ | Foundation: upload, extraction, parsing, dashboard |
| 2 | Job description analysis and transparent match scoring (matched / missing skills) |
| 3 | CV improvement recommendations, content-weakness checks, downloadable reports |
| 4 | Lightweight semantic similarity, synonyms, contextual skill extraction, optional free/local AI |
| 5 | Hardening, edge-case testing, performance review, final deployment polish |

## License

Choose a license for your repository (for example MIT) before publishing.
