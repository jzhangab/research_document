from __future__ import annotations

from dataclasses import dataclass

import requests


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
    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str, max_results: int = 8):
        if not api_key:
            raise ValueError("SERPAPI_KEY is required")
        self._api_key = api_key
        self._max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        params = self._build_params(query)
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
        except requests.RequestException as e:
            raise SerpAPIError(f"Network error: {e}") from e

        if resp.status_code != 200:
            raise SerpAPIError(f"SerpAPI returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()

        if "error" in data:
            raise SerpAPIError(f"SerpAPI error: {data['error']}")

        results: list[SearchResult] = []
        for item in data.get("organic_results", [])[: self._max_results]:
            from urllib.parse import urlparse
            domain = urlparse(item.get("link", "")).netloc
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source=domain,
                    position=item.get("position", 0),
                )
            )

        return results

    def _build_params(self, query: str) -> dict:
        return {
            "q": query,
            "api_key": self._api_key,
            "engine": "google",
            "num": self._max_results,
            "hl": "en",
            "gl": "us",
        }
