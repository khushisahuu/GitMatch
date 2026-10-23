import os
import time
import requests
import pandas as pd

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    raise SystemExit("GITHUB_TOKEN is not set. See the setup step.")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

LANGUAGES = ["python", "javascript", "typescript", "java", "go"]
PAGES_PER_LANGUAGE = 2  # 100 issues per page

rows = []
for lang in LANGUAGES:
    for page in range(1, PAGES_PER_LANGUAGE + 1):
        params = {
            "q": f'label:"good first issue" state:open is:issue language:{lang}',
            "sort": "updated",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }
        r = requests.get("https://api.github.com/search/issues",
                         headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f"Problem on {lang} page {page}: {r.status_code} {r.text[:200]}")
            break
        for item in r.json().get("items", []):
            rows.append({
                "title": item["title"],
                "body": (item.get("body") or "")[:1000],
                "labels": ", ".join(l["name"] for l in item["labels"]),
                "repo": item["repository_url"].replace("https://api.github.com/repos/", ""),
                "language": lang,
                "url": item["html_url"],
                "comments": item["comments"],
                "created_at": item["created_at"],
            })
        print(f"{lang} page {page}: total so far {len(rows)}")
        time.sleep(2)  # be polite to the API

df = pd.DataFrame(rows).drop_duplicates(subset="url")
df.to_csv("issues.csv", index=False)
print(f"\nSaved {len(df)} issues to issues.csv")
