# CreatorLens AI 2.0

A Multilingual Social Media Intelligence and Audience Understanding System.
A Streamlit prototype that turns English, Tamil and Tanglish (Romanised Tamil) comments into ranked content ideas, FAQ clusters and a review queue.

## Features

- Dark enterprise dashboard (charcoal `#0B0B0C`, coral `#FF3E30` accents, rounded cards)
- **Module 1:** text normalisation, emoji feature extraction, language tagging (English / Tamil / Code-Mixed)
- **Module 2:** sentiment and intent classifier (heuristic stand-in for XLM-RoBERTa / MuRIL)
- **Module 3:** semantic grouping and FAQ extraction (character n-gram TF-IDF + cosine similarity)
- **Module 4:** opportunity ranker: `Score = (similar requests × 1.5) + (avg likes × 0.8) + structural severity weight`
- Flagged review queue and CSV export

## Project structure

```
creatorlens-ai/
├── app.py                  # entire application
├── requirements.txt
├── .streamlit/config.toml  # dark theme
├── .vscode/                # VS Code settings + debug launcher
└── .gitignore
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

In VS Code: open the folder, select the `.venv` interpreter, then press **F5** (`CreatorLens AI: Streamlit`).

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: CreatorLens AI 2.0 prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/creatorlens-ai.git
git push -u origin main
```

## Deploy (optional)

Push the repo to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.

## Notes

- All data is synthetic (`get_mock_data()`). Replace it with real YouTube / Instagram comment exports.
- To go production-grade, replace `classify_comment()` with a multilingual transformer.
