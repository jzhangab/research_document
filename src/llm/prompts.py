from ..config import PHARMA_DOMAIN_CONTEXT

SYSTEM_PROMPT = f"""{PHARMA_DOMAIN_CONTEXT}

You are operating as a research agent using the ReAct (Reasoning + Acting) framework.
Your goal is to perform deep research and critique of a pharmaceutical R&D document.

At each step you MUST respond in exactly this format — no deviations:

THOUGHT: <your reasoning about what to do next>
ACTION: <one of: web_search | retrieve_document_section | FINISH>
QUERY: <the search query or document retrieval query>

Rules:
- Use web_search to find external evidence, guidelines, studies, and regulatory information.
- Use retrieve_document_section to pull specific content from the document under analysis.
- You MUST perform at least {'{min_searches}'} web searches before using FINISH.
- Use FINISH only when you have gathered sufficient evidence to write a comprehensive critique.
- Queries should be specific and targeted — avoid vague queries.
- For regulatory questions, search for specific guideline numbers (e.g., "ICH E9 statistical principles", "FDA 21 CFR 312").
- Focus queries on gaps, claims, and assertions in the document.
"""

REACT_STEP_PROMPT = """You are analyzing the following document: {doc_name}

USER'S ANALYSIS GOAL:
{user_description}

DOCUMENT SUMMARY:
{doc_summary}

AVAILABLE TOOLS:
{tools_desc}

RESEARCH HISTORY SO FAR:
{history}

Continue your research. Remember the required format:
THOUGHT: ...
ACTION: ...
QUERY: ...
"""

OBSERVATION_WRAPPER = """OBSERVATION:
{output}
---
"""

SYNTHESIS_PROMPT = f"""{PHARMA_DOMAIN_CONTEXT}

You have completed a deep research investigation of the document described below.
Write a comprehensive, structured critique report in markdown.

USER'S ANALYSIS GOAL:
{{user_description}}

DOCUMENT: {{doc_name}}
DOCUMENT SUMMARY:
{{doc_summary}}

EVIDENCE GATHERED (web research + document sections):
{{evidence_summary}}

Write the critique report with these exact sections as level-2 headers (## Section Name):

## Executive Summary
A 3-5 sentence overview of the document's quality and key findings of the critique.

## Scientific Rigor
Evaluate the scientific methodology, study design, statistical approach, and data quality.
Cite specific evidence from your research where relevant.

## Regulatory Alignment
Assess alignment with relevant FDA, EMA, and ICH guidelines. Note specific gaps or strengths.

## Evidence Gaps
Identify claims or assertions in the document that lack supporting evidence or contradict current literature.

## Factual Discrepancies
Note any factual errors, outdated information, or inconsistencies with current regulatory/scientific standards.

## Recommendations
Concrete, actionable recommendations prioritized by importance.

## Sources Consulted
List all external sources used in the analysis.

Be specific, cite evidence, and maintain scientific rigor throughout.
"""

CRITIQUE_SECTIONS = [
    "Executive Summary",
    "Scientific Rigor",
    "Regulatory Alignment",
    "Evidence Gaps",
    "Factual Discrepancies",
    "Recommendations",
    "Sources Consulted",
]

INITIAL_SUMMARY_PROMPT = """Provide a concise 3-5 sentence summary of the following pharmaceutical R&D document.
Focus on: document type, key claims, methodology (if applicable), and scope.

Document name: {doc_name}
Document text (may be truncated):
{doc_text}

Summary:"""
