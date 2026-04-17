from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from ..config import Config
from ..llm.mesh_client import MeshClient
from ..llm.prompts import SYNTHESIS_PROMPT, CRITIQUE_SECTIONS
from ..document.parser import ParsedDocument
from .react_loop import AgentState
from .tools import ToolResult, ToolName


@dataclass
class CritiqueReport:
    executive_summary: str = ""
    scientific_rigor: str = ""
    regulatory_alignment: str = ""
    evidence_gaps: str = ""
    factual_discrepancies: str = ""
    recommendations: str = ""
    sources_consulted: list[str] = field(default_factory=list)
    raw_markdown: str = ""


class Synthesizer:
    def __init__(self, llm: MeshClient, config: Config):
        self._llm = llm
        self._config = config

    def synthesize(self, state: AgentState, parsed_doc: ParsedDocument) -> CritiqueReport:
        messages = self._build_messages(state, parsed_doc)
        raw = self._llm.complete(messages)
        return self._build_report(raw, state)

    def synthesize_stream(
        self, state: AgentState, parsed_doc: ParsedDocument
    ) -> Iterator[str]:
        messages = self._build_messages(state, parsed_doc)
        buffer = []
        for chunk in self._llm.complete_stream(messages):
            buffer.append(chunk)
            yield chunk
        raw = "".join(buffer)
        self._last_report = self._build_report(raw, state)

    def _build_messages(self, state: AgentState, parsed_doc: ParsedDocument) -> list[dict]:
        evidence_summary = self._build_evidence_summary(state.tool_results)
        prompt = SYNTHESIS_PROMPT.format(
            user_description=state.user_description,
            doc_name=state.doc_name,
            doc_summary=state.doc_summary,
            evidence_summary=evidence_summary,
        )
        return [{"role": "user", "text": prompt}]

    def _build_evidence_summary(self, results: list[ToolResult]) -> str:
        if not results:
            return "No external evidence gathered."

        web_results = [r for r in results if r.tool_name == ToolName.WEB_SEARCH and r.success]
        doc_results = [r for r in results if r.tool_name == ToolName.RETRIEVE_DOC and r.success]

        parts: list[str] = []

        if web_results:
            parts.append("=== WEB RESEARCH FINDINGS ===")
            for i, r in enumerate(web_results, 1):
                parts.append(f"\nSearch {i}: \"{r.query}\"\n{r.output[:1500]}")

        if doc_results:
            parts.append("\n=== DOCUMENT SECTIONS RETRIEVED ===")
            for i, r in enumerate(doc_results, 1):
                parts.append(f"\nRetrieval {i}: \"{r.query}\"\n{r.output[:1500]}")

        return "\n".join(parts)

    def _build_report(self, raw_markdown: str, state: AgentState) -> CritiqueReport:
        sections = self._parse_sections(raw_markdown)
        urls = self._extract_urls(state)

        return CritiqueReport(
            executive_summary=sections.get("Executive Summary", ""),
            scientific_rigor=sections.get("Scientific Rigor", ""),
            regulatory_alignment=sections.get("Regulatory Alignment", ""),
            evidence_gaps=sections.get("Evidence Gaps", ""),
            factual_discrepancies=sections.get("Factual Discrepancies", ""),
            recommendations=sections.get("Recommendations", ""),
            sources_consulted=urls,
            raw_markdown=raw_markdown,
        )

    def _parse_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        pattern = r"##\s+(.+?)\n(.*?)(?=\n##\s|\Z)"
        for match in re.finditer(pattern, text, re.DOTALL):
            title = match.group(1).strip()
            body = match.group(2).strip()
            sections[title] = body
        if not sections:
            sections["Executive Summary"] = text
        return sections

    def _extract_urls(self, state: AgentState) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for result in state.tool_results:
            if result.tool_name == ToolName.WEB_SEARCH and result.raw:
                for sr in result.raw:
                    if sr.url and sr.url not in seen:
                        urls.append(sr.url)
                        seen.add(sr.url)
        return urls
