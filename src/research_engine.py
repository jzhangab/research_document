from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .config import Config
from .document.parser import DocumentParser, ParsedDocument
from .document.chunker import Chunker
from .document.retriever import HybridRetriever
from .llm.mesh_client import MeshClient
from .llm.prompts import INITIAL_SUMMARY_PROMPT
from .search.serp_client import SerpClient
from .agent.tools import ToolRegistry
from .agent.react_loop import ReactLoop, AgentState
from .agent.synthesizer import Synthesizer, CritiqueReport


class ResearchCritiqueSystem:
    """Facade: wires all modules together. The only class the notebook needs."""

    def __init__(self, config: Config):
        self._config = config
        self._llm = MeshClient(config.llm_id, config.dataiku_project_key)
        self._serp = SerpClient(config.serp_api_key, config.serp_max_results)
        self._parser = DocumentParser()
        self._chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self._synthesizer = Synthesizer(self._llm, config)
        self._last_report: CritiqueReport | None = None

    def run(self, file_path: str, description: str) -> CritiqueReport:
        parsed_doc, state = self._prepare(file_path, description)
        loop = self._make_loop(parsed_doc)
        loop.run(state)
        report = self._synthesizer.synthesize(state, parsed_doc)
        self._last_report = report
        return report

    def run_stream(self, file_path: str, description: str) -> Iterator[str]:
        """
        Yields progress strings during the ReAct loop, then markdown tokens
        during synthesis. Suitable for live display in the notebook.
        """
        parsed_doc, state = self._prepare(file_path, description)
        loop = self._make_loop(parsed_doc)

        yield f"**Document loaded:** {parsed_doc.file_name}\n\n"
        yield f"**Summary:** {state.doc_summary}\n\n---\n\n"
        yield "**Starting research agent...**\n\n"

        for step in loop.run_stream(state):
            if step.is_final:
                yield f"**Step {step.iteration} — Research complete. Writing critique...**\n\n"
            else:
                tool_label = step.action_tool.value if step.action_tool else "unknown"
                yield (
                    f"**Step {step.iteration}** `{tool_label}`: {step.action_query}\n"
                    f"> {step.thought[:200]}{'...' if len(step.thought) > 200 else ''}\n\n"
                )

        yield "---\n\n## Critique Report\n\n"
        for chunk in self._synthesizer.synthesize_stream(state, parsed_doc):
            yield chunk

        self._last_report = self._synthesizer._last_report

    def list_available_llms(self) -> list[str]:
        return MeshClient.list_llm_ids(self._config.dataiku_project_key)

    def _prepare(self, file_path: str, description: str) -> tuple[ParsedDocument, AgentState]:
        parsed_doc = self._parser.parse(file_path)
        chunks = self._chunker.chunk(parsed_doc)
        retriever = HybridRetriever(chunks, self._config.top_k_retrieval)
        retriever.build_index()
        self._retriever = retriever

        doc_summary = self._summarize_doc(parsed_doc)

        state = AgentState(
            user_description=description,
            doc_name=parsed_doc.file_name,
            doc_summary=doc_summary,
        )
        return parsed_doc, state

    def _make_loop(self, parsed_doc: ParsedDocument) -> ReactLoop:
        tools = ToolRegistry(self._serp, self._retriever)
        return ReactLoop(self._llm, tools, self._config)

    def _summarize_doc(self, parsed_doc: ParsedDocument) -> str:
        # Use first ~3000 chars for the summary prompt to keep it cheap
        excerpt = parsed_doc.raw_text[:3000]
        prompt = INITIAL_SUMMARY_PROMPT.format(
            doc_name=parsed_doc.file_name,
            doc_text=excerpt,
        )
        return self._llm.complete([{"role": "user", "text": prompt}])
