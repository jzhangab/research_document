from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from ..config import Config
from ..llm.mesh_client import MeshClient
from ..llm.prompts import REACT_STEP_PROMPT, SYSTEM_PROMPT, OBSERVATION_WRAPPER
from .tools import ToolRegistry, ToolResult, ToolName


@dataclass
class AgentStep:
    iteration: int
    thought: str
    action_tool: ToolName | None
    action_query: str | None
    observation: str | None
    is_final: bool = False


@dataclass
class AgentState:
    user_description: str
    doc_name: str
    doc_summary: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


class ReactLoop:
    def __init__(self, llm: MeshClient, tools: ToolRegistry, config: Config):
        self._llm = llm
        self._tools = tools
        self._config = config

    def run(self, state: AgentState) -> AgentState:
        for step in self.run_stream(state):
            pass  # consume iterator; state is mutated in place
        return state

    def run_stream(self, state: AgentState) -> Iterator[AgentStep]:
        search_count = 0

        for iteration in range(1, self._config.max_iterations + 1):
            messages = self._build_messages(state)
            raw_output = self._llm.complete(messages)

            thought, tool_name, query, is_final = self._parse_llm_output(raw_output)

            # Enforce minimum searches before allowing FINISH
            if is_final and search_count < self._config.min_searches_required:
                thought += (
                    f" [Agent note: minimum {self._config.min_searches_required} "
                    f"web searches required; only {search_count} completed so far. Continuing research.]"
                )
                is_final = False
                # Re-parse as a continue signal — generate a follow-up search
                tool_name = ToolName.WEB_SEARCH
                query = f"{state.user_description} pharmaceutical evidence"

            step = AgentStep(
                iteration=iteration,
                thought=thought,
                action_tool=tool_name,
                action_query=query,
                observation=None,
                is_final=is_final,
            )

            if is_final:
                state.steps.append(step)
                yield step
                break

            if tool_name is None:
                step.observation = "ERROR: Could not parse a valid ACTION. Please respond with THOUGHT/ACTION/QUERY format."
            else:
                result = self._tools.execute(tool_name, query)
                state.tool_results.append(result)
                if tool_name == ToolName.WEB_SEARCH and result.success:
                    search_count += 1

                if result.success:
                    step.observation = OBSERVATION_WRAPPER.format(output=result.output)
                else:
                    step.observation = f"OBSERVATION: Tool error — {result.error}. Try a different query or tool."

            state.steps.append(step)
            yield step

        return state

    def _build_messages(self, state: AgentState) -> list[dict]:
        system_text = SYSTEM_PROMPT.replace(
            "{min_searches}", str(self._config.min_searches_required)
        )
        user_text = REACT_STEP_PROMPT.format(
            doc_name=state.doc_name,
            user_description=state.user_description,
            doc_summary=state.doc_summary,
            tools_desc=self._tools.describe(),
            history=self._format_history(state.steps),
        )
        return [
            {"role": "system", "text": system_text},
            {"role": "user", "text": user_text},
        ]

    def _parse_llm_output(
        self, text: str
    ) -> tuple[str, ToolName | None, str | None, bool]:
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION:|\Z)", text, re.DOTALL)
        action_match = re.search(r"ACTION:\s*(\S+)", text)
        query_match = re.search(r"QUERY:\s*(.+?)(?=\n[A-Z]+:|\Z)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else text.strip()
        action_raw = action_match.group(1).strip().lower() if action_match else ""
        query = query_match.group(1).strip() if query_match else ""

        if "finish" in action_raw:
            return thought, None, None, True

        tool_name: ToolName | None = None
        if "web_search" in action_raw:
            tool_name = ToolName.WEB_SEARCH
        elif "retrieve" in action_raw:
            tool_name = ToolName.RETRIEVE_DOC

        return thought, tool_name, query if query else None, False

    def _format_history(self, steps: list[AgentStep]) -> str:
        if not steps:
            return "No steps taken yet. Begin your research."
        parts: list[str] = []
        for step in steps:
            entry = f"Step {step.iteration}:\n  THOUGHT: {step.thought}"
            if step.action_tool:
                entry += f"\n  ACTION: {step.action_tool.value}"
                entry += f"\n  QUERY: {step.action_query}"
            if step.observation:
                # Truncate long observations to avoid blowing context
                obs = step.observation
                if len(obs) > 1200:
                    obs = obs[:1200] + "\n... [truncated]"
                entry += f"\n  {obs}"
            parts.append(entry)
        return "\n\n".join(parts)
