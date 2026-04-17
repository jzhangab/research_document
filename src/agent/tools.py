from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..search.serp_client import SerpClient, SearchResult, SerpAPIError
from ..document.retriever import HybridRetriever
from ..document.chunker import Chunk


class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    RETRIEVE_DOC = "retrieve_document_section"


@dataclass
class ToolResult:
    tool_name: ToolName
    query: str
    output: str
    raw: Any
    success: bool
    error: str | None = None


class ToolRegistry:
    def __init__(self, serp_client: SerpClient, retriever: HybridRetriever):
        self._serp = serp_client
        self._retriever = retriever

    def web_search(self, query: str) -> ToolResult:
        try:
            results = self._serp.search(query)
            if not results:
                return ToolResult(
                    tool_name=ToolName.WEB_SEARCH,
                    query=query,
                    output="No results found for this query.",
                    raw=[],
                    success=True,
                )
            lines = []
            for r in results:
                lines.append(f"[{r.position}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
            return ToolResult(
                tool_name=ToolName.WEB_SEARCH,
                query=query,
                output="\n\n".join(lines),
                raw=results,
                success=True,
            )
        except SerpAPIError as e:
            return ToolResult(
                tool_name=ToolName.WEB_SEARCH,
                query=query,
                output="",
                raw=None,
                success=False,
                error=str(e),
            )

    def retrieve_document_section(self, query: str) -> ToolResult:
        try:
            chunks = self._retriever.retrieve(query)
            if not chunks:
                return ToolResult(
                    tool_name=ToolName.RETRIEVE_DOC,
                    query=query,
                    output="No relevant sections found.",
                    raw=[],
                    success=True,
                )
            parts = []
            for chunk in chunks:
                parts.append(f"[Page {chunk.page_number}]\n{chunk.text}")
            return ToolResult(
                tool_name=ToolName.RETRIEVE_DOC,
                query=query,
                output="\n\n---\n\n".join(parts),
                raw=chunks,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name=ToolName.RETRIEVE_DOC,
                query=query,
                output="",
                raw=None,
                success=False,
                error=str(e),
            )

    def execute(self, tool_name: ToolName, query: str) -> ToolResult:
        if tool_name == ToolName.WEB_SEARCH:
            return self.web_search(query)
        elif tool_name == ToolName.RETRIEVE_DOC:
            return self.retrieve_document_section(query)
        return ToolResult(
            tool_name=tool_name,
            query=query,
            output="",
            raw=None,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    @staticmethod
    def describe() -> str:
        return (
            "web_search(query): Search the web via Google for external evidence, "
            "guidelines, publications, and regulatory documents.\n"
            "retrieve_document_section(query): Retrieve relevant sections from the "
            "document under analysis using semantic + keyword search."
        )
