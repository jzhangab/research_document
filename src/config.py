import os
from dataclasses import dataclass, field
from typing import Optional

PHARMA_DOMAIN_CONTEXT = (
    "You are an expert pharmaceutical R&D analyst with deep expertise in "
    "clinical trial design, regulatory submissions (FDA/EMA/ICH guidelines), "
    "drug development pipelines, biostatistics, pharmacokinetics, and CMC. "
    "You evaluate documents with scientific rigor and regulatory awareness."
)

@dataclass
class Config:
    llm_id: str = "gpt-4o"
    dataiku_project_key: Optional[str] = None

    chunk_size: int = 600
    chunk_overlap: int = 80
    top_k_retrieval: int = 5

    serp_api_key: str = field(default_factory=lambda: os.environ.get("SERPAPI_KEY", ""))
    serp_max_results: int = 8

    max_iterations: int = 12
    min_searches_required: int = 3
