from __future__ import annotations

from typing import Iterator


class MeshClient:
    """Wraps Dataiku LLM Mesh. All `dataiku` imports are isolated here."""

    def __init__(self, llm_id: str, project_key: str | None = None):
        import dataiku

        client = dataiku.api_client()
        key = project_key or dataiku.default_project_key()
        self._project = client.get_project(key)
        self._llm_id = llm_id

    def complete(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "text": str}, ...]
        Returns full response string.
        """
        llm = self._project.get_llm(self._llm_id)
        completion = llm.new_completion()
        for msg in messages:
            completion.with_message(msg["text"], role=msg["role"])
        resp = completion.execute()
        return resp.text

    def complete_stream(self, messages: list[dict]) -> Iterator[str]:
        """Yields text chunks. Falls back to complete() if streaming unsupported."""
        try:
            llm = self._project.get_llm(self._llm_id)
            completion = llm.new_completion()
            for msg in messages:
                completion.with_message(msg["text"], role=msg["role"])
            for chunk in completion.execute_stream():
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except (AttributeError, TypeError):
            # Model or Mesh version doesn't support streaming
            yield self.complete(messages)

    @classmethod
    def list_llm_ids(cls, project_key: str | None = None) -> list[str]:
        """Returns available LLM IDs from the Dataiku project."""
        import dataiku

        client = dataiku.api_client()
        key = project_key or dataiku.default_project_key()
        project = client.get_project(key)
        try:
            return [llm["id"] for llm in project.list_llms()]
        except Exception:
            return []
