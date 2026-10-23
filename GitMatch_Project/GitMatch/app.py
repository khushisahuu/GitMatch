"""Web server for the issue recommender.

Run:  uvicorn app:app --reload
Open: http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from recommender import IssueRecommender, fetch_github_user_profile

BASE = Path(__file__).parent
REAL_CSV = BASE / "issues.csv"
SAMPLE_CSV = BASE / "sample_issues.csv"

# Use the real data if issues.csv is next to this file, else fall back to the sample
USING_SAMPLE = not REAL_CSV.exists()
CSV_PATH = SAMPLE_CSV if USING_SAMPLE else REAL_CSV

engine = IssueRecommender(CSV_PATH)
print(f"Loaded {len(engine.df)} issues from {CSV_PATH.name}")
if USING_SAMPLE:
    print("NOTE: issues.csv not found, running on the built-in sample data.")

app = FastAPI(title="GitMatch: Intelligent Open-Source Issue Recommender")


@app.get("/api/stats")
def stats():
    return engine.stats(using_sample=USING_SAMPLE, source=CSV_PATH.name)


@app.get("/api/recommend")
def recommend(
    q: str = Query(..., min_length=1, max_length=300),
    top_n: int = Query(5, ge=1, le=20),
    languages: str = Query("", max_length=100),
    experience_level: str = Query("beginner", max_length=20),
    liked_urls: str = Query("", max_length=3000),
    disliked_urls: str = Query("", max_length=3000),
    live: bool = Query(False),
):
    langs = [l for l in languages.split(",") if l.strip()]
    liked = [u.strip() for u in liked_urls.split(",") if u.strip()]
    disliked = [u.strip() for u in disliked_urls.split(",") if u.strip()]
    
    return engine.recommend(
        skills=q,
        top_n=top_n,
        languages=langs,
        experience_level=experience_level.lower(),
        liked_urls=liked,
        disliked_urls=disliked,
        live=live,
    )


@app.get("/api/github-user")
def get_github_user(username: str = Query(..., min_length=1, max_length=100)):
    profile = fetch_github_user_profile(username)
    if not profile:
        return {"error": f"Could not find public repos for user '{username}'"}
    return profile


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
