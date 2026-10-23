"""Issue Recommender Engine: Hybrid Content-Based Filtering with Concept Expansion.

Key Components for Recommender Systems:
1. Concept & Domain Skill Expansion: Resolves vocabulary mismatch (e.g. 'pandas' -> 'dataframe', 'series').
2. Bigram TF-IDF Representation: Captures multi-word phrases (e.g. 'machine learning', 'data cleaning').
3. Multi-Attribute Utility (Hybrid Scoring):
   Final Score = w1 * TextSimilarity + w2 * CoreSkillMatch + w3 * UnclaimedBonus + w4 * ExperienceFit
4. Explainable AI (XAI): Returns explicit "Why Recommended" justifications per issue.
5. Interactive Relevance Feedback: Session-based like/dislike adjustments (Rocchio algorithm inspiration).
"""

import ast
import json
import os
import re
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Language mapping aliases
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

# Domain knowledge graph / concept expansion dictionary.
# This solves the vocabulary gap (e.g., matching "pandas" with issues talking about "dataframes").
SKILL_EXPANSIONS = {
    "pandas": ["pandas", "dataframe", "dataframes", "series", "read_csv", "data analysis"],
    "numpy": ["numpy", "ndarray", "matrix", "matrices", "linear algebra"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn", "model", "training", "dataset", "classification", "regression"],
    "ml": ["machine learning", "ml", "scikit-learn", "sklearn", "model", "training", "dataset"],
    "deep learning": ["deep learning", "neural network", "pytorch", "torch", "tensorflow", "keras"],
    "nlp": ["nlp", "natural language", "token", "tokenizer", "embeddings", "bert", "transformers"],
    "data cleaning": ["data cleaning", "clean", "missing values", "null", "imputation", "parser", "preprocessing"],
    "sql": ["sql", "database", "query", "postgres", "mysql", "sqlite", "table", "schema"],
    "django": ["django", "orm", "models", "migration", "views", "urls"],
    "fastapi": ["fastapi", "pydantic", "endpoint", "route", "api", "uvicorn", "async"],
    "flask": ["flask", "blueprint", "route", "endpoint", "api"],
    "react": ["react", "jsx", "tsx", "component", "hooks", "state", "props", "frontend"],
    "frontend": ["frontend", "ui", "css", "html", "style", "component", "layout"],
    "backend": ["backend", "api", "server", "database", "endpoint", "controller", "rest"],
    "docker": ["docker", "container", "dockerfile", "compose"],
    "api": ["api", "rest", "endpoint", "crud", "http", "json", "request"],
    "testing": ["test", "testing", "pytest", "unittest", "mock", "coverage"],
    "documentation": ["documentation", "docs", "readme", "typo", "guide", "tutorial"],
}

# Common Spanish / Portuguese words to filter out international issues that slip past ASCII checks
NON_ENGLISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "para", "por",
    "con", "como", "mas", "pero", "este", "esta", "estos", "estas", "não", "uma", "para", "com",
    "seu", "sua", "você", "isso", "pelo", "pela"
}

ENGLISH_COMMON_WORDS = {
    "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "with", "this",
    "that", "issue", "fix", "add", "error", "update", "support", "create", "bug", "feature"
}

COLUMNS = ["title", "body", "labels", "language", "repo", "url", "comments", "created_at"]


def is_english(text):
    """Accurate English detector: Combines ASCII character ratio with stop-word verification.
    
    Prevents Romance languages (Spanish, Portuguese, etc.) from slipping through.
    """
    if not text or len(text.strip()) < 10:
        return True
    
    text_lower = text.lower()
    
    # 1. Non-ASCII symbol ratio test
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if (non_ascii / len(text)) > 0.12:
        return False
        
    # 2. Vocabulary stop-word ratio test
    words = re.findall(r"\b[a-z]{2,}\b", text_lower)
    if not words:
        return True
        
    non_eng_count = sum(1 for w in words if w in NON_ENGLISH_STOPWORDS)
    eng_count = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)
    
    # If it contains noticeable Spanish/Portuguese markers and very few English markers, filter it
    if non_eng_count >= 3 and non_eng_count > eng_count:
        return False
        
    return True


def detect_languages(skills):
    """Find any programming languages mentioned in the user's input."""
    words = re.findall(r"[a-z]+", skills.lower())
    return {LANG_ALIASES[w] for w in words if w in LANG_ALIASES}


def parse_labels(raw):
    """Safely extracts GitHub labels from list strings, JSON or CSV."""
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


def make_snippet(body, limit=220):
    """Cleans markdown noise, URLs, and HTML layout tags for a crisp issue preview."""
    text = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)   # images
    text = re.sub(r"https?://\S+", " ", text)             # urls
    text = re.sub(r"```.*?```", " ", text, flags=re.S)    # code blocks
    text = re.sub(
        r"</?(?:br|p|div|span|details|summary|img|a|ul|ol|li|h[1-6]|b|i|em"
        r"|strong|code|pre|table|tr|td|th|thead|tbody|hr|sub|sup|kbd)\b[^>]*>",
        " ", text, flags=re.I,
    )
    text = re.sub(r"[#`*_<>\[\]|]|-{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def fetch_live_github_issues(query, languages=None, max_count=12):
    """Fetches real-time open 'good first issue' tickets directly from GitHub REST API."""
    token = os.environ.get("GITHUB_TOKEN")
    
    q_parts = ['label:"good first issue"', 'state:open', 'is:issue']
    if languages:
        clean_langs = [l.strip().lower() for l in languages if l.strip()]
        if clean_langs:
            q_parts.append(f"language:{clean_langs[0]}")
    if query:
        clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
        words = [w for w in clean_q.split() if len(w) >= 3 and w.lower() not in ENGLISH_COMMON_WORDS]
        if words:
            q_parts.append(" ".join(words[:3]))

    search_str = " ".join(q_parts)
    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(search_str)}&sort=updated&order=desc&per_page={max_count}"
    
    headers = {
        "User-Agent": "GitMatch-IssueRecommender",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
                rows = []
                for item in items:
                    repo_name = item.get("repository_url", "").replace("https://api.github.com/repos/", "")
                    labels_list = [l.get("name", "") for l in item.get("labels", []) if isinstance(l, dict)]
                    lang = (languages[0] if languages else "python").lower()
                    
                    rows.append({
                        "title": item.get("title", ""),
                        "body": (item.get("body") or "")[:1000],
                        "labels": ", ".join(labels_list),
                        "parsed_labels": labels_list,
                        "parsed_labels_lower": [l.lower() for l in labels_list],
                        "language": lang,
                        "lang_key": lang,
                        "repo": repo_name,
                        "url": item.get("html_url", ""),
                        "comments": int(item.get("comments", 0)),
                        "created_at": item.get("created_at", ""),
                        "is_live": True,
                    })
                return rows
    except Exception as e:
        print(f"Notice: Live GitHub fetch encountered: {e}")
        return []
    return []


# Recognized technology stacks and frameworks to prevent personal/private project names
# (e.g. "stylematch", "assignment1") from polluting the recommendation queries.
RECOGNIZED_TECH_KEYWORDS = {
    "python", "javascript", "typescript", "java", "go", "golang", "c++", "rust",
    "react", "vue", "angular", "nextjs", "nodejs", "express", "django", "fastapi", "flask",
    "spring", "pandas", "numpy", "scikit-learn", "sklearn", "pytorch", "tensorflow", "keras",
    "machine-learning", "machine learning", "ml", "deep-learning", "nlp", "ai",
    "data-science", "data analysis", "data cleaning", "backend", "frontend", "fullstack",
    "api", "rest", "graphql", "sql", "database", "docker", "kubernetes", "devops",
    "testing", "pytest", "documentation", "microservices", "web", "cli",
}

DEFAULT_STACK_KEYWORDS = {
    "python": ["API", "Backend"],
    "javascript": ["React", "Frontend"],
    "typescript": ["Frontend", "API"],
    "java": ["Backend", "API"],
    "go": ["Backend", "CLI"],
}


def filter_tech_topics(raw_terms):
    """Filters out arbitrary repository names and keeps only genuine technology/domain topics."""
    valid = []
    for term in raw_terms:
        clean = re.sub(r"[^a-zA-Z0-9-]", " ", str(term)).strip().lower()
        parts = clean.replace("-", " ").split()
        # Direct match or multi-word match
        if clean in RECOGNIZED_TECH_KEYWORDS or clean.replace("-", " ") in RECOGNIZED_TECH_KEYWORDS:
            if clean not in valid:
                valid.append(clean.title())
        else:
            for part in parts:
                if part in RECOGNIZED_TECH_KEYWORDS and part not in valid:
                    valid.append(part.title())
    return valid


def fetch_github_user_profile(username):
    """Fetches public repositories of a user to infer their tech stack and primary languages.
    
    Filters out personal project titles and only extracts verified technologies and frameworks.
    Automatically falls back to public profile web scraping if the GitHub API rate limit is reached.
    """
    username = (username or "").strip().lstrip("@")
    if not username:
        return None
        
    token = os.environ.get("GITHUB_TOKEN")
    url = f"https://api.github.com/users/{urllib.parse.quote(username)}/repos?per_page=30&sort=updated"
    headers = {
        "User-Agent": "GitMatch-IssueRecommender",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                repos = json.loads(resp.read().decode("utf-8"))
                lang_counts = {}
                raw_topics = []
                for r in repos:
                    lang = (r.get("language") or "").strip().lower()
                    if lang and lang in LANG_ALIASES:
                        canonical = LANG_ALIASES[lang]
                        lang_counts[canonical] = lang_counts.get(canonical, 0) + 1
                    for t in r.get("topics", []):
                        if t:
                            raw_topics.append(t)
                    r_name = (r.get("name") or "").strip().lower()
                    if r_name and r_name != username.lower():
                        raw_topics.append(r_name)
                
                sorted_langs = [l for l, _ in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)]
                filtered_topics = filter_tech_topics(raw_topics)
                
                # If no specific framework tag was found, infer appropriate stack from primary language
                if not filtered_topics and sorted_langs:
                    primary_lang = sorted_langs[0]
                    filtered_topics = DEFAULT_STACK_KEYWORDS.get(primary_lang, ["API", "Backend"])

                return {
                    "username": username,
                    "languages": sorted_langs[:3],
                    "topics": filtered_topics[:6],
                    "repo_count": len(repos),
                }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"User '@{username}' does not exist on GitHub."}
        print(f"Notice: GitHub REST API returned {e.code} ({e.reason}). Using public profile fallback.")
    except Exception as e:
        print(f"Notice: GitHub API fetch encountered: {e}. Using public profile fallback.")

    # Fallback: Scrape public repositories web page (exempt from REST API rate limit)
    try:
        web_url = f"https://github.com/{urllib.parse.quote(username)}?tab=repositories"
        web_req = urllib.request.Request(web_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        })
        with urllib.request.urlopen(web_req, timeout=6) as resp:
            if resp.status == 200:
                html = resp.read().decode("utf-8", errors="ignore")
                raw_langs = re.findall(r'itemprop=["\']programmingLanguage["\']>([^<]+)<', html)
                repos = [
                    r.strip() for r in re.findall(r'itemprop=["\']name codeRepository["\'][^>]*>\s*([^\s<]+)', html)
                    if r.strip() and r.strip().lower() != username.lower()
                ]
                topics = [
                    t.strip() for t in re.findall(r'class=["\']topic-tag[^>]*>\s*([^\s<]+)', html)
                    if t.strip()
                ]

                # Combine topics & repo names for tech keyword filtering
                combined_raw = topics + repos
                filtered_topics = filter_tech_topics(combined_raw)

                langs = []
                for l in raw_langs:
                    l_lower = l.strip().lower()
                    canonical = LANG_ALIASES.get(l_lower, l_lower)
                    if canonical and canonical not in langs:
                        langs.append(canonical)

                # Default fallback stack if personal repo name was filtered out
                if not filtered_topics and langs:
                    primary_lang = langs[0]
                    filtered_topics = DEFAULT_STACK_KEYWORDS.get(primary_lang, ["API", "Backend"])

                return {
                    "username": username,
                    "languages": langs[:3],
                    "topics": filtered_topics[:6],
                    "repo_count": len(repos) or len(raw_langs) or 1,
                }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"User '@{username}' does not exist on GitHub."}
        print(f"Notice: Web fallback encountered {e.code}: {e.reason}")
    except Exception as e:
        print(f"Notice: Web fallback encountered: {e}")

    return None


class IssueRecommender:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).fillna("")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = "" if col != "comments" else 0
            if col == "comments":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = df[col].astype(str)

        # Filter out non-English content
        sample = df["title"] + " " + df["body"].str[:300]
        df = df[sample.apply(is_english)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No usable issues found in {csv_path}")

        df["lang_key"] = df["language"].str.lower().str.strip()

        # Parse labels once for fast multi-criteria scoring
        df["parsed_labels"] = df["labels"].apply(parse_labels)
        df["parsed_labels_lower"] = df["parsed_labels"].apply(lambda lst: [l.lower() for l in lst])

        # Feature Engineering:
        # Title is given 4x weight, labels 3x weight, language 2x weight over body text.
        df["text"] = (
            (df["title"] + " ") * 4
            + (df["labels"] + " ") * 3
            + (df["language"] + " ") * 2
            + df["body"]
        )

        self.df = df
        
        # Bigram TF-IDF vectorizer (captures multi-word phrases like 'machine learning')
        # sublinear_tf=True dampens dominating high-frequency generic terms
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True
        )
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

    def expand_query(self, user_skills):
        """Domain Concept Expansion: Expands query with related technical terms.
        
        Example: 'python pandas' -> 'python pandas dataframe series read_csv pd'
        """
        user_skills_lower = user_skills.lower()
        expanded_tokens = [user_skills_lower]
        detected_concepts = set()

        for concept, synonyms in SKILL_EXPANSIONS.items():
            # Check whole word / phrase match
            pattern = r"\b" + re.escape(concept) + r"\b"
            if re.search(pattern, user_skills_lower):
                detected_concepts.add(concept)
                expanded_tokens.extend(synonyms)

        expanded_text = " ".join(expanded_tokens)
        return expanded_text, detected_concepts

    def calculate_skill_overlap(self, row, detected_concepts, user_skills_terms):
        """Calculates precise match bonus for specific technologies in title or labels.
        
        Uses regex word boundaries so short tokens (e.g. 'ui', 'ai') never match
        as substrings inside unrelated words (e.g. 'build' or 'train').
        """
        title_lower = row["title"].lower()
        labels_lower = row["parsed_labels_lower"]
        matched_skills = set()

        # Check detected high-level concepts using word boundaries
        for concept in detected_concepts:
            synonyms = SKILL_EXPANSIONS.get(concept, [concept])
            concept_found = False
            for syn in synonyms:
                pattern = r"\b" + re.escape(syn.lower()) + r"\b"
                if re.search(pattern, title_lower):
                    matched_skills.add(concept)
                    concept_found = True
                    break
                if any(re.search(pattern, lbl) for lbl in labels_lower):
                    matched_skills.add(concept)
                    concept_found = True
                    break

        # Check raw user skill words (e.g. custom libraries)
        for term in user_skills_terms:
            if len(term) >= 3 and term not in ENGLISH_COMMON_WORDS:
                pattern = r"\b" + re.escape(term.lower()) + r"\b"
                if re.search(pattern, title_lower) or any(re.search(pattern, lbl) for lbl in labels_lower):
                    matched_skills.add(term)

        # Normalize score
        if not detected_concepts and not user_skills_terms:
            return 0.5, []
            
        target_count = len(detected_concepts) if detected_concepts else len(user_skills_terms)
        score = min(1.0, len(matched_skills) / max(1, target_count))
        return score, sorted(matched_skills)

    def calculate_experience_fit(self, row, experience_level):
        """Scores alignment with the user's experience level (Beginner vs Intermediate vs Advanced).
        
        Specifically treats 'good first issue' differently per level:
        - Beginner: 'good first issue' and docs/typo/starter = Strong Boost (1.0)
        - Intermediate: 'good first issue' = Neutral; real code tasks (bug, enhancement, feature, refactor) = Boost (1.0); pure typos = Demoted (0.3)
        - Advanced: 'good first issue' and simple typo/docs = Penalty/Suppressed (0.0); architecture, performance, security, core = Strong Boost (1.0)
        """
        labels = row["parsed_labels_lower"]
        title_lower = row["title"].lower()
        
        is_trivial_task = any(
            any(w in lbl for w in ["typo", "readme", "documentation", "docs"])
            for lbl in labels
        ) or any(w in title_lower for w in ["readme", "typo", "fix typo", "update readme", "update config"])

        has_good_first_issue = any("good first issue" in lbl for lbl in labels)

        is_code_task = any(
            any(w in lbl for w in ["enhancement", "feature", "refactor", "bug", "help wanted", "test", "fix"])
            for lbl in labels
        ) or any(w in title_lower for w in ["fix", "add", "implement", "support", "endpoint", "function", "bug", "reject"])

        is_advanced_task = any(
            any(w in lbl for w in ["performance", "security", "core", "architecture", "refactor", "concurrency", "optimization", "schema", "database"])
            for lbl in labels
        ) or any(w in title_lower for w in ["performance", "optimize", "architecture", "concurrency", "security", "leak", "pipeline", "parser"])

        if experience_level == "beginner":
            if is_trivial_task or has_good_first_issue:
                return 1.0
            return 0.5

        elif experience_level == "intermediate":
            if is_code_task and not is_trivial_task:
                return 1.0
            elif is_trivial_task:
                return 0.25  # Demote pure typos for intermediate devs
            return 0.6

        elif experience_level == "advanced":
            if is_advanced_task:
                return 1.0
            elif is_trivial_task:
                return 0.05  # Heavily suppress simple typo fixes for advanced devs
            elif has_good_first_issue and not is_code_task:
                return 0.20
            return 0.5

        return 0.6

    def calculate_unclaimed_bonus(self, comments_count):
        """Gives a priority boost to issues with 0 or 1 comments (unclaimed issues)."""
        if comments_count == 0:
            return 1.0
        elif comments_count <= 2:
            return 0.6
        else:
            return 0.2

    def recommend(
        self,
        skills,
        top_n=5,
        languages=None,
        experience_level="beginner",
        liked_urls=None,
        disliked_urls=None,
        live=True,
    ):
        """Hybrid Recommender System.
        
        Combines:
        1. TF-IDF Cosine Similarity on expanded concept text.
        2. Core Skill/Technology Overlap (Title + Label verification).
        3. Unclaimed Priority (Low comment count bonus).
        4. Experience Level Fit (Adaptive multi-criteria weighting).
        5. Interactive Feedback Re-ranking (Rocchio relevance reinforcement).
        6. On-Demand Live GitHub Streaming option.
        """
        skills = (skills or "").strip()
        liked_urls = set(liked_urls or [])
        disliked_urls = set(disliked_urls or [])

        # Filter languages: Explicitly chosen take priority, otherwise detect from input
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
            "experience_level": experience_level,
            "source": "live_github" if live else "indexed",
            "results": [],
        }

        if not skills:
            return response

        # Adaptive Multi-Criteria Weights per Persona:
        # Beginner values unclaimed tickets and beginner tags.
        # Intermediate values specific library/tool match and code tasks.
        # Advanced values complex domain tasks and text depth.
        if experience_level == "beginner":
            w_sim, w_skill, w_unclaimed, w_exp = 0.35, 0.20, 0.25, 0.20
        elif experience_level == "intermediate":
            w_sim, w_skill, w_unclaimed, w_exp = 0.45, 0.30, 0.10, 0.15
        elif experience_level == "advanced":
            w_sim, w_skill, w_unclaimed, w_exp = 0.50, 0.25, 0.00, 0.25
        else:
            w_sim, w_skill, w_unclaimed, w_exp = 0.40, 0.25, 0.15, 0.20

        # Extract traits of liked issues for Rocchio profile refinement
        liked_repos = set()
        liked_labels = set()
        if liked_urls:
            for idx in range(len(self.df)):
                r_url = self.df.iloc[idx]["url"]
                if r_url in liked_urls:
                    liked_repos.add(self.df.iloc[idx]["repo"])
                    liked_labels.update(self.df.iloc[idx]["parsed_labels_lower"])

        # Step 1: Query Expansion (synonym and concept mapping)
        expanded_query, detected_concepts = self.expand_query(skills)
        user_vector = self.vectorizer.transform([expanded_query])
        
        # Step 2: Content-based Cosine Similarity
        cosine_scores = cosine_similarity(user_vector, self.issue_vectors).flatten()

        # Step 3: Hard Language Masking (if languages are specified)
        if langs:
            lang_mask = self.df["lang_key"].isin(langs).values
            cosine_scores = cosine_scores * lang_mask

        # Step 4: Extract meaningful query terms for overlap checks
        user_terms = set(self.feature_names[user_vector.nonzero()[1]])
        core_query_words = set(re.findall(r"\b[a-z]{3,}\b", skills.lower())) - ENGLISH_COMMON_WORDS

        # Step 5: Hybrid Multi-Attribute Utility Scoring
        hybrid_scores = np.zeros(len(self.df))
        reasons_map = {}
        matched_skills_map = {}
        breakdown_map = {}

        for i in range(len(self.df)):
            if cosine_scores[i] <= 0:
                continue

            row = self.df.iloc[i]
            url = row["url"]

            # Filter out explicitly disliked/dismissed issues
            if url in disliked_urls:
                continue

            sim_score = float(cosine_scores[i])
            skill_score, matched_skills = self.calculate_skill_overlap(row, detected_concepts, core_query_words)
            unclaimed_score = self.calculate_unclaimed_bonus(row["comments"])
            exp_score = self.calculate_experience_fit(row, experience_level)

            # Penalty if generic match only (e.g. user asked for pandas, but neither pandas nor dataframe matched)
            if detected_concepts and skill_score == 0:
                sim_score *= 0.35

            # Dynamic Adaptive Multi-attribute combination
            combined = (
                w_sim * sim_score
                + w_skill * skill_score
                + w_unclaimed * unclaimed_score
                + w_exp * exp_score
            )

            # Rocchio Relevance Feedback:
            # Boost other issues from the same repository or matching liked labels
            reasons = []
            if url in liked_urls:
                combined += 0.25
                reasons.append("★ Saved by You")
            elif row["repo"] in liked_repos:
                combined += 0.14
                reasons.append("Boosted (liked repo)")
            elif liked_labels and any(lbl in liked_labels for lbl in row["parsed_labels_lower"]):
                combined += 0.08
                reasons.append("Boosted (liked tech)")

            hybrid_scores[i] = combined
            breakdown_map[i] = {
                "text_sim": round(float(sim_score), 2),
                "skill_match": round(float(skill_score), 2),
                "unclaimed": round(float(unclaimed_score), 2),
                "exp_fit": round(float(exp_score), 2),
                "weights": {
                    "text": int(w_sim * 100),
                    "skill": int(w_skill * 100),
                    "unclaimed": int(w_unclaimed * 100),
                    "exp": int(w_exp * 100),
                }
            }

            # Generate Explainable AI (XAI) rationale
            if matched_skills:
                reasons.append(f"Matches skill: {', '.join(matched_skills)}")
            elif row["lang_key"]:
                reasons.append(f"Matches {DISPLAY_LANG.get(row['lang_key'], row['lang_key'].title())}")
                
            if row["comments"] == 0 and experience_level == "beginner":
                reasons.append("Unclaimed (0 comments)")
            elif row["comments"] <= 2 and experience_level != "advanced":
                reasons.append("Low competition")

            if experience_level == "beginner" and exp_score >= 0.8:
                reasons.append("Beginner-friendly label")
            elif experience_level == "intermediate" and exp_score >= 0.8:
                reasons.append("Code implementation task")
            elif experience_level == "advanced" and exp_score >= 0.8:
                reasons.append("Advanced task")

            reasons_map[i] = reasons
            matched_skills_map[i] = matched_skills

        # Check for Live GitHub issues if requested
        live_cards = []
        if live:
            raw_live = fetch_live_github_issues(skills, languages=list(langs), max_count=12)
            if raw_live:
                live_texts = [
                    ((r["title"] + " ") * 4) + ((r["labels"] + " ") * 3) + ((r["language"] + " ") * 2) + r["body"]
                    for r in raw_live
                ]
                try:
                    live_vecs = self.vectorizer.transform(live_texts)
                    live_sims = cosine_similarity(user_vector, live_vecs).flatten()
                    for idx, r in enumerate(raw_live):
                        if r["url"] in disliked_urls:
                            continue
                        l_sim = float(live_sims[idx])
                        l_skill, l_matched = self.calculate_skill_overlap(r, detected_concepts, core_query_words)
                        l_unclaimed = self.calculate_unclaimed_bonus(r["comments"])
                        l_exp = self.calculate_experience_fit(r, experience_level)
                        if detected_concepts and l_skill == 0:
                            l_sim *= 0.35
                        l_comb = (w_sim * l_sim) + (w_skill * l_skill) + (w_unclaimed * l_unclaimed) + (w_exp * l_exp)
                        l_reasons = ["🟢 Live from GitHub (Just Now)"]
                        if l_matched:
                            l_reasons.append(f"Matches skill: {', '.join(l_matched)}")
                        if r["comments"] == 0:
                            l_reasons.append("Unclaimed (0 comments)")
                        live_cards.append({
                            "title": r["title"],
                            "repo": r["repo"],
                            "language": DISPLAY_LANG.get(r["lang_key"], r["language"].title()),
                            "language_key": r["lang_key"],
                            "score": round(float(l_comb), 3),
                            "matched_words": sorted(user_terms)[:6],
                            "matched_skills": l_matched,
                            "reasons": l_reasons,
                            "is_liked": r["url"] in liked_urls,
                            "is_live": True,
                            "comments": r["comments"],
                            "created_at": r["created_at"],
                            "labels": [l for l in r["parsed_labels"] if l.lower() != "good first issue"][:5],
                            "snippet": make_snippet(r["body"]),
                            "url": r["url"],
                            "score_breakdown": {
                                "text_sim": round(float(l_sim), 2),
                                "skill_match": round(float(l_skill), 2),
                                "unclaimed": round(float(l_unclaimed), 2),
                                "exp_fit": round(float(l_exp), 2),
                                "weights": {
                                    "text": int(w_sim * 100),
                                    "skill": int(w_skill * 100),
                                    "unclaimed": int(w_unclaimed * 100),
                                    "exp": int(w_exp * 100),
                                }
                            }
                        })
                except Exception as e:
                    print(f"Notice: Live scoring error: {e}")

        # Step 6: Rank and deduplicate by repository (1 issue per repo)
        top = []
        seen_repos = set()
        
        # Incorporate live cards first if present
        for card in live_cards:
            if card["repo"] not in seen_repos:
                seen_repos.add(card["repo"])
                response["results"].append(card)

        for i in hybrid_scores.argsort()[::-1]:
            if hybrid_scores[i] <= 0 or len(response["results"]) >= top_n + len(live_cards):
                break
            repo = self.df.iloc[i]["repo"]
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            top.append(i)

        for rank, i in enumerate(top, len(live_cards) + 1):
            row = self.df.iloc[i]
            issue_terms = set(self.feature_names[self.issue_vectors[i].nonzero()[1]])
            shared_words = sorted((user_terms & issue_terms) - ENGLISH_COMMON_WORDS)
            
            labels = [
                l for l in row["parsed_labels"]
                if l.lower() != "good first issue"
            ]

            response["results"].append({
                "rank": rank,
                "title": row["title"],
                "repo": row["repo"],
                "language": DISPLAY_LANG.get(row["lang_key"], row["language"]),
                "language_key": row["lang_key"],
                "score": round(float(hybrid_scores[i]), 3),
                "matched_words": shared_words[:8],
                "matched_skills": matched_skills_map.get(i, []),
                "reasons": reasons_map.get(i, []),
                "is_liked": row["url"] in liked_urls,
                "is_live": False,
                "comments": int(row["comments"]),
                "created_at": row["created_at"],
                "labels": labels[:5],
                "snippet": make_snippet(row["body"]),
                "url": row["url"],
                "score_breakdown": breakdown_map.get(i, {}),
            })

        # Calculate relative scores and assign sequential ranks
        best_score = max((r["score"] for r in response["results"]), default=1.0)
        for idx, r in enumerate(response["results"], 1):
            r["rank"] = idx
            r["relative"] = round(float(r["score"]) / max(best_score, 0.001), 3)

        return response
