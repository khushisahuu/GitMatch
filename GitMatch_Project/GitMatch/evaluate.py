r"""Evaluation Script for Issue Finder Recommender System.

Demonstrates quantitative offline evaluation using classic Information Retrieval (IR)
and Recommender System metrics:
1. Precision@K: Fraction of top-K recommended issues that match the user persona's target domain.
2. Hit Rate@K: Binary indicator of whether at least one relevant recommendation was made in top-K.
3. Mean Reciprocal Rank (MRR): 1 / rank of the first relevant recommendation (evaluates ranking quality).

Run:
    ..\.venv\Scripts\python.exe evaluate.py
"""

import sys
from recommender import IssueRecommender

# Define evaluation personas with explicit domain criteria
TEST_PERSONAS = [
    {
        "name": "Persona 1: Python Backend / API Dev",
        "query": "python api backend endpoint rest",
        "language": ["python"],
        "experience": "intermediate",
        "relevant_keywords": ["api", "backend", "endpoint", "rest", "server", "crud", "http", "route"],
        "target_language": "python"
    },
    {
        "name": "Persona 2: Frontend React Engineer",
        "query": "typescript react frontend component ui",
        "language": ["typescript", "javascript"],
        "experience": "beginner",
        "relevant_keywords": ["react", "frontend", "ui", "component", "css", "html", "style"],
        "target_language": ["typescript", "javascript"]
    },
    {
        "name": "Persona 3: Python Data / ML Student",
        "query": "python machine learning model dataset",
        "language": ["python"],
        "experience": "beginner",
        "relevant_keywords": ["model", "learning", "data", "dataset", "ai", "train", "classification", "pipeline"],
        "target_language": "python"
    },
    {
        "name": "Persona 4: Go Systems Developer",
        "query": "go backend cli microservice",
        "language": ["go"],
        "experience": "intermediate",
        "relevant_keywords": ["cli", "backend", "server", "command", "api", "microservice", "service"],
        "target_language": "go"
    },
    {
        "name": "Persona 5: First-Time Open Source Contributor",
        "query": "python documentation typo readme guide",
        "language": ["python"],
        "experience": "beginner",
        "relevant_keywords": ["documentation", "docs", "readme", "typo", "guide", "tutorial", "doc"],
        "target_language": "python"
    }
]


def is_relevant(issue, persona):
    """Ground truth relevance rule:
    An issue is relevant if its language matches and at least one core domain keyword appears
    in its title, labels, or repository.
    """
    lang_match = False
    target = persona["target_language"]
    if isinstance(target, list):
        lang_match = issue["language_key"] in target
    else:
        lang_match = issue["language_key"] == target

    if not lang_match:
        return False

    content_to_check = (
        issue["title"].lower()
        + " " + " ".join(issue["labels"]).lower()
        + " " + issue["repo"].lower()
    )

    keyword_hit = any(kw in content_to_check for kw in persona["relevant_keywords"])
    return keyword_hit


def evaluate_recommender(csv_path="issues.csv", top_k=5):
    print("=" * 70)
    print("       GITMATCH - RECOMMENDER SYSTEM EVALUATION REPORT")
    print("=" * 70)

    try:
        engine = IssueRecommender(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return

    precision_scores = []
    hit_scores = []
    reciprocal_ranks = []

    print(f"\nEvaluating Top-{top_k} recommendations on {len(TEST_PERSONAS)} benchmark student personas:\n")

    for p in TEST_PERSONAS:
        resp = engine.recommend(
            skills=p["query"],
            top_n=top_k,
            languages=p["language"],
            experience_level=p["experience"]
        )

        results = resp["results"][:top_k]
        hits = 0
        first_hit_rank = 0

        for idx, issue in enumerate(results, start=1):
            if is_relevant(issue, p):
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = idx

        prec = hits / top_k if top_k > 0 else 0.0
        hit = 1.0 if hits > 0 else 0.0
        rr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

        precision_scores.append(prec)
        hit_scores.append(hit)
        reciprocal_ranks.append(rr)

        print(f"- {p['name']}")
        print(f"  Precision@{top_k}: {prec:.2f} ({hits}/{top_k} relevant) | First relevant rank: #{first_hit_rank or 'None'} (RR: {rr:.2f})")

    avg_prec = sum(precision_scores) / len(precision_scores)
    avg_hit = sum(hit_scores) / len(hit_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 70)
    print("                    AGGREGATE SUMMARY METRICS")
    print("=" * 70)
    print(f"  * Mean Precision@{top_k}       : {avg_prec * 100:.1f}%")
    print(f"  * Hit Rate@{top_k}             : {avg_hit * 100:.1f}%")
    print(f"  * Mean Reciprocal Rank (MRR) : {mrr:.3f}")
    print("=" * 70)
    print("Interpretation:")
    print(" - Precision@5 measures the purity of the top results.")
    print(" - Hit Rate@5 confirms every persona finds at least 1 viable match in their top 5.")
    print(" - MRR near 1.0 indicates the #1 or #2 ranked issue is almost always a direct hit.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_recommender()
