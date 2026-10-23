"""Baseline Issue recommender: TF-IDF + cosine similarity (original v1).

Preserved as a safety fallback.
"""

import ast
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Words a user might type -> the language name used in the CSV (lowercase)
LANG_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java",
    "go": "go", "golang": "go",
}

DISPLAY_LANG = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "go": "Go",
}

COLUMNS = ["title", "body", "labels", "language", "repo", "url"]


def is_english(text):
    """True if less than 10% of the characters are non-English symbols."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) < 0.1


def detect_languages(skills):
    """Find any programming languages mentioned in the user's skills."""
    words = re.findall(r"[a-z]+", skills.lower())
    return {LANG_ALIASES[w] for w in words if w in LANG_ALIASES}


def parse_labels(raw):
    """Labels may be stored as "a, b", "['a', 'b']" or JSON. Return a list."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                return [str(x).strip() for x in loader(raw) if str(x).strip()]
            except (ValueError, SyntaxError):
                continue
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_snippet(body, limit=240):
    """A short, readable preview of an issue body (markdown noise removed)."""
    text = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)   # images
    text = re.sub(r"https?://\S+", " ", text)             # urls
    text = re.sub(r"```.*?```", " ", text, flags=re.S)    # code blocks
    text = re.sub(                                        # html layout tags
        r"</?(?:br|p|div|span|details|summary|img|a|ul|ol|li|h[1-6]|b|i|em"
        r"|strong|code|pre|table|tr|td|th|thead|tbody|hr|sub|sup|kbd)\b[^>]*>",
        " ", text, flags=re.I,
    )
    text = re.sub(r"[#`*_<>\[\]|]|-{3,}", " ", text)      # leftover symbols, table lines
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


class IssueRecommender:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).fillna("")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str)

        # Drop issues that are mostly non-English
        sample = df["title"] + " " + df["body"].str[:300]
        df = df[sample.apply(is_english)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No usable issues found in {csv_path}")

        df["lang_key"] = df["language"].str.lower().str.strip()

        # Title is repeated 3x so it counts more than the body
        df["text"] = (
            (df["title"] + " ") * 3
            + df["labels"] + " "
            + df["language"] + " "
            + df["body"]
        )

        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.issue_vectors = self.vectorizer.fit_transform(df["text"])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def stats(self, using_sample, source):
        counts = self.df["lang_key"].value_counts()
        languages = [
            {
                "key": key,
                "name": DISPLAY_LANG.get(key, key.title()),
                "count": int(count),
            }
            for key, count in counts.items()
            if key
        ]
        return {
            "total_issues": int(len(self.df)),
            "languages": languages,
            "using_sample": using_sample,
            "source": source,
        }

    def recommend(self, skills, top_n=5, languages=None):
        skills = (skills or "").strip()

        # Languages ticked in the UI win; otherwise detect them from the text
        if languages:
            langs = {
                LANG_ALIASES.get(l.lower().strip(), l.lower().strip())
                for l in languages
                if l.strip()
            }
            auto_detected = False
        else:
            langs = detect_languages(skills)
            auto_detected = bool(langs)

        response = {
            "query": skills,
            "languages": sorted(langs),
            "auto_detected": auto_detected,
            "results": [],
        }
        if not skills:
            return response

        user_vector = self.vectorizer.transform([skills])
        scores = cosine_similarity(user_vector, self.issue_vectors).flatten()

        # If languages are set, zero out issues from other languages
        if langs:
            mask = self.df["lang_key"].isin(langs).values
            scores = scores * mask

        user_terms = set(self.feature_names[user_vector.nonzero()[1]])

        # Best to worst, keeping only one issue per repo
        top = []
        seen_repos = set()
        for i in scores.argsort()[::-1]:
            if scores[i] <= 0 or len(top) == top_n:
                break
            repo = self.df.iloc[i]["repo"]
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            top.append(i)

        best = float(scores[top[0]]) if top else 1.0
        for rank, i in enumerate(top, 1):
            row = self.df.iloc[i]
            issue_terms = set(self.feature_names[self.issue_vectors[i].nonzero()[1]])
            labels = [
                l for l in parse_labels(row["labels"])
                if l.lower() != "good first issue"
            ]
            response["results"].append({
                "rank": rank,
                "title": row["title"],
                "repo": row["repo"],
                "language": DISPLAY_LANG.get(row["lang_key"], row["language"]),
                "language_key": row["lang_key"],
                "score": round(float(scores[i]), 3),
                "relative": round(float(scores[i]) / best, 3),
                "matched_words": sorted(user_terms & issue_terms),
                "labels": labels[:4],
                "snippet": make_snippet(row["body"]),
                "url": row["url"],
            })
        return response
