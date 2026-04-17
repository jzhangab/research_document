from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    position: int


class SerpAPIError(Exception):
    pass


class SerpClient:
    def __init__(self, api_key: str, max_results: int = 8):
        if not api_key:
            raise ValueError("SERPAPI_KEY is required")
        self._api_key = api_key
        self._max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        try:
            from serpapi import GoogleSearch
        except ImportError:
            raise SerpAPIError(
                "serpapi package not installed. Run: pip install google-search-results"
            )

        try:
            data = GoogleSearch({
                "q": query,
                "api_key": self._api_key,
                "engine": "google",
                "num": self._max_results,
                "hl": "en",
                "gl": "us",
            }).get_dict()
        except Exception as e:
            raise SerpAPIError(f"SerpAPI call failed: {e}") from e

        if "error" in data:
            raise SerpAPIError(f"SerpAPI error: {data['error']}")

        results: list[SearchResult] = []
        for item in data.get("organic_results", [])[: self._max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source=urlparse(item.get("link", "")).netloc,
                    position=item.get("position", 0),
                )
            )

        return results
