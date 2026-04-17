import hashlib
from dataclasses import dataclass
from typing import Sequence

from .parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    char_start: int
    char_end: int
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sent_tokenize(text: str) -> list[str]:
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]


class Chunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        doc_hash = hashlib.md5(doc.file_name.encode()).hexdigest()[:8]

        for page_num, page_text in doc.page_map.items():
            sentences = _sent_tokenize(page_text)
            page_chunks = self._build_chunks(sentences, page_num, page_text, doc_hash)
            chunks.extend(page_chunks)

        return chunks

    def _build_chunks(
        self,
        sentences: list[str],
        page_num: int,
        page_text: str,
        doc_hash: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_tokens = 0
        chunk_idx = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = _estimate_tokens(sent)

            if buf_tokens + sent_tokens > self.chunk_size and buffer:
                chunk_text = " ".join(buffer)
                char_start = page_text.find(buffer[0])
                char_end = char_start + len(chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=f"doc_{doc_hash}_p{page_num}_{chunk_idx}",
                        text=chunk_text,
                        page_number=page_num,
                        char_start=max(0, char_start),
                        char_end=max(0, char_end),
                        token_estimate=buf_tokens,
                    )
                )
                chunk_idx += 1

                # overlap: keep last N tokens worth of sentences
                overlap_buffer: list[str] = []
                overlap_tokens = 0
                for s in reversed(buffer):
                    t = _estimate_tokens(s)
                    if overlap_tokens + t > self.overlap:
                        break
                    overlap_buffer.insert(0, s)
                    overlap_tokens += t
                buffer = overlap_buffer
                buf_tokens = overlap_tokens
            else:
                buffer.append(sent)
                buf_tokens += sent_tokens
                i += 1

        if buffer:
            chunk_text = " ".join(buffer)
            char_start = page_text.find(buffer[0])
            char_end = char_start + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=f"doc_{doc_hash}_p{page_num}_{chunk_idx}",
                    text=chunk_text,
                    page_number=page_num,
                    char_start=max(0, char_start),
                    char_end=max(0, char_end),
                    token_estimate=buf_tokens,
                )
            )

        return chunks
