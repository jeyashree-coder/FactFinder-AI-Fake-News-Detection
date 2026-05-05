from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.database import insert_bbc_article

BBC_HOME = "https://www.bbc.com/news"


def safe_get(url: str, timeout: int = 15) -> requests.Response:
    response = requests.get(
        url,
        headers={"User-Agent": "FactFinderBot/1.0 (+https://www.bbc.com/news)"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def normalize_bbc_link(href: str) -> str | None:
    if not href:
        return None
    if href.startswith("https://www.bbc.com/news/articles/"):
        return href
    if href.startswith("/news/articles/"):
        return f"https://www.bbc.com{href}"
    return None


def parse_article(url: str) -> dict[str, Any] | None:
    response = safe_get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("h1")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p")]
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("main p")]

    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    content = " ".join(paragraphs).strip()
    if not title or len(content) < 80:
        return None

    published_date = None
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        published_date = time_tag["datetime"]
    else:
        last_modified = response.headers.get("Last-Modified")
        if last_modified:
            try:
                published_date = parsedate_to_datetime(last_modified).isoformat()
            except (TypeError, ValueError):
                published_date = None

    return {
        "title": title,
        "content": content,
        "published_date": published_date,
        "source": "BBC News",
        "url": url,
    }


def collect_bbc_news() -> dict[str, Any]:
    collected_at = datetime.now().isoformat()
    inserted = 0
    skipped = 0
    errors: list[str] = []

    try:
        soup = BeautifulSoup(safe_get(BBC_HOME).text, "html.parser")
        candidate_urls = []
        seen = set()
        for link in soup.select("a[href]"):
            normalized = normalize_bbc_link(link.get("href"))
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidate_urls.append(normalized)
            if len(candidate_urls) >= 15:
                break

        for url in candidate_urls:
            try:
                article = parse_article(url)
                if not article:
                    skipped += 1
                    continue
                was_inserted = insert_bbc_article(
                    title=article["title"],
                    content=article["content"],
                    published_date=article["published_date"],
                    source=article["source"],
                    collected_at=collected_at,
                    url=article["url"],
                )
                if was_inserted:
                    inserted += 1
                else:
                    skipped += 1
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")

        return {
            "status": "completed",
            "inserted": inserted,
            "skipped": skipped,
            "checked_urls": len(candidate_urls),
            "collected_at": collected_at,
            "errors": errors,
        }
    except requests.RequestException as exc:
        return {
            "status": "failed",
            "inserted": 0,
            "skipped": 0,
            "checked_urls": 0,
            "collected_at": collected_at,
            "errors": [str(exc)],
        }
