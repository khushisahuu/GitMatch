# GitMatch: Intelligent Open-Source Issue Recommender

An explainable, multi-attribute recommender system that matches developers and students to open-source GitHub issues ("good first issue") tailored to their technical skills, target languages, and experience level.

---

## Key Highlights & Recommender Architecture

1. **Content-Based Filtering with Domain Concept Expansion:**
   - Solves vocabulary mismatch (e.g. searching `pandas` expands to technical synsets: `dataframe`, `series`, `read_csv`, `data analysis`).
   - Uses **Bigram TF-IDF** `(ngram_range=(1, 2))` with logarithmic term damping (`sublinear_tf=True`) so composite concepts like `"machine learning"` or `"data cleaning"` are recognized as single entities rather than splitting into ambiguous unigrams.

2. **Multi-Attribute Utility Function (Hybrid Scoring):**
   Recommendations are ranked using a multi-criteria scoring formula:
   $$\text{FinalScore} = 0.50 \cdot \text{CosineSimilarity} + 0.25 \cdot \text{CoreSkillOverlap} + 0.15 \cdot \text{UnclaimedPriority} + 0.10 \cdot \text{ExperienceFit}$$
   - **Core Skill Gating:** Strongly penalizes issues that only match incidental stop-adjacent words like *"data"* without actually matching core technical concepts.
   - **Unclaimed Priority:** Rewards issues with 0 or few comments so new contributors don't pick already-claimed tickets.
   - **Experience Fit:** Adjusts scoring weights for **Beginner** (prioritizes documentation, typo, beginner tags) vs. **Intermediate** (prioritizes enhancements, features, refactoring).

3. **Interactive Relevance Feedback Loop:**
   - Real-time session-based re-ranking inspired by the **Rocchio algorithm**.
   - Clicking **"Interested"** reinforces positive candidate weights; clicking **"Dismiss"** filters out tickets and dampens negative signals.

4. **Explainable AI (XAI):**
   - Each card displays explicit transparent rationale badges (*"Why Recommended: Matches Python & Pandas • Unclaimed (0 comments) • Beginner-friendly"*).

5. **Quantitative Offline Evaluation (`evaluate.py`):**
   - Benchmarked across 5 standard student personas.
   - **Mean Precision@5:** 88.0%
   - **Hit Rate@5:** 100.0%
   - **Mean Reciprocal Rank (MRR):** 0.867

---

## Quickstart

Run with the project's dedicated virtual environment:

```bash
# 1. Start the web server
..\.venv\Scripts\python.exe -m uvicorn app:app --reload

# 2. Open the web interface
http://127.0.0.1:8000

# 3. Run the evaluation benchmark
..\.venv\Scripts\python.exe evaluate.py
```

---

## File Structure

| File | Purpose |
| --- | --- |
| [`recommender.py`](file:///c:/KHUSHI/Khushi%20Study/PPTS/3rd%20year/RECOMMENDER%20SYS/issue-finder/issue-finder/recommender.py) | Upgraded Hybrid Recommender engine (Bigrams, Concept Expansion, Multi-Attribute Scoring, XAI reasons) |
| [`app.py`](file:///c:/KHUSHI/Khushi%20Study/PPTS/3rd%20year/RECOMMENDER%20SYS/issue-finder/issue-finder/app.py) | FastAPI service exposing `/api/recommend` and `/api/stats` with session feedback support |
| [`evaluate.py`](file:///c:/KHUSHI/Khushi%20Study/PPTS/3rd%20year/RECOMMENDER%20SYS/issue-finder/issue-finder/evaluate.py) | IR & RecSys evaluation harness measuring Precision@5, Hit Rate@5, and MRR |
| [`static/index.html`](file:///c:/KHUSHI/Khushi%20Study/PPTS/3rd%20year/RECOMMENDER%20SYS/issue-finder/issue-finder/static/index.html) | Nord / Cyber Frost UI featuring user persona controls, XAI badges, and feedback loop |
| `recommender_v1_baseline.py` | Safety fallback copy of the original baseline engine |
| `static/index_v1_baseline.html`| Safety fallback copy of the original UI |
| `collect_issues.py` | Ingestion pipeline fetching open issues from GitHub REST API |
| `issues.csv` | Dataset of ~1,000 GitHub issues across Python, JS, TS, Java, and Go |
