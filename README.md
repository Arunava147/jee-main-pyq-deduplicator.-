# JEE Main PYQ Deduplicator

A local web app prototype that:
- accepts a PYQ PDF,
- extracts question text and page images,
- uses Gemini to identify question groups,
- automatically determines the number of representative questions,
- shows frequency counts and lets you inspect the original questions grouped with each representative,
- exports CSV and Excel reports.

## Requirements
- Python 3.10 or newer
- A Gemini API key from Google AI Studio
- Internet connection for Gemini API calls

## Run on Windows
1. Install Python from https://www.python.org/downloads/ (enable "Add Python to PATH").
2. Extract this ZIP.
3. Open Command Prompt in the extracted folder.
4. Run:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. A local web page should open. Paste your Gemini API key in the sidebar, upload a PDF, and click "Extract, deduplicate and build report".

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes and limitations
- This prototype does not require you to choose a target number of unique questions. It keeps one representative per group detected by the model.
- "Frequency" counts the representative itself plus all questions grouped with it.
- "View repeats" shows the actual grouped question text and source references.
- PDF pages are sent as images and extracted text to the Gemini API for analysis. Do not upload confidential material.
- Question extraction and semantic grouping can make mistakes, particularly with low-resolution scans, diagrams, and ambiguous problem variants. Review the grouping before studying from the exported set.
- If processing fails on a large PDF, try a smaller PDF or reduce "Pages per extraction batch".
- Model availability and API pricing/quotas can change; check Google's current AI Studio documentation before use.
