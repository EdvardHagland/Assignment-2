"""Prompt templates for Gemini qualitative interpretation.

This prompt layer is intentionally comparative and critical.

The model is not being asked to "bless" the clusters. It is being asked to
review the clustering result as an exploratory analysis bundle and say what the
partition appears to capture, where the clusters overlap, and whether some of
the apparent differences may be weak or artifactual.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


SYSTEM_PROMPT = """You are a critical qualitative reviewer of an exploratory text-clustering result.

Role boundaries:
- You are not discovering clusters. The clusters already exist.
- You are not writing a domain essay. You are reviewing the supplied analysis bundle.
- You must stay strictly grounded in the provided tables, cluster statistics, contrastive terms, validation summaries, and representative text units.

What the input means:
- Each cluster contains real text units selected from the analysis pipeline.
- The text units may be paragraph-like chunks rather than complete responses.
- The contrastive terms are post hoc lexical descriptors computed after clustering. They are clues, not proof on their own.
- Metadata signals report overrepresentation relative to the corpus baseline. Treat them as descriptive lift signals, not causes.
- Representative examples may include both central examples and variation examples. Use the roles to judge cluster coherence and internal spread.
- The cluster statistics describe scale, response coverage, and source mix. Use them to understand breadth and composition, not to invent causal stories.
- The global tables describe the full clustering result, including model-selection evidence and overlap across clusters.

Writing rules:
- Prefer neutral, analytical labels over rhetorical or activist phrasing.
- Infer topic language only from the supplied bundle. Do not add outside policy knowledge.
- If two or more clusters seem to be weak subdivisions of a broader discourse, say so explicitly.
- If the result looks partly artifactual or over-partitioned, say so explicitly.
- Do not reduce paragraph-level semantic clusters to mere keyword lists.
- Use the representative text units as the main evidence and use the terms only as supporting cues.
- When comparing clusters, focus on what most distinguishes one cluster from the others, even if that difference is only marginal.
- If the terms are noisy or boilerplate-like, rely more on the representative text units and say less, not more.
- Do not infer anything from fields that are not present.
- Return valid JSON only. No markdown, no code fences, no commentary."""


def _truncate_text(value: Any, limit: int = 900) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_batch_prompt(
    batch_id: str,
    clusters: Sequence[Mapping[str, Any]],
    *,
    analysis_bundle: Mapping[str, Any] | None = None,
    corpus_name: str,
    research_question: str,
) -> str:
    """Render a structured prompt for a batch of clusters."""

    payload = {
        "batch_id": batch_id,
        "corpus_name": corpus_name,
        "research_question": research_question,
        "analysis_context": {
            "task": (
                "Interpret and critically review existing embedding-based clusters as recurring "
                "semantic frames using only the supplied analysis outputs."
            ),
            "report_contract": (
                "The report separately shows the cluster statistics, validation summary, top-term "
                "table, and representative evidence. Your output should synthesize those signals rather "
                "than restating every number."
            ),
            "term_method": (
                "Top terms were generated after clustering with a contrastive lexical-signal step. "
                "Treat them as descriptive clues, not definitive proof."
            ),
            "unit_note": (
                "Representative examples may be paragraph-like units extracted from longer "
                "responses, so labels should describe recurring frames rather than whole-document identities."
            ),
            "critical_note": (
                "A good answer may conclude that the clusters are only weakly differentiated, partly overlapping, "
                "or best interpreted as sub-variations of a broader discourse."
            ),
        },
        "global_tables": analysis_bundle or {},
        "clusters": [],
    }

    for cluster in clusters:
        payload["clusters"].append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "cluster_statistics": cluster.get("cluster_statistics", cluster.get("cluster_stats", {})),
                "contrastive_terms": cluster.get("contrastive_terms", cluster.get("top_terms", [])),
                "metadata_signals": cluster.get("metadata_signals", []),
                "representative_examples": [
                    {
                        "source_id": ex.get("source_id"),
                        "response_key": ex.get("response_key"),
                        "parent_document_id": ex.get("parent_document_id"),
                        "segment_index": ex.get("segment_index"),
                        "segment_count": ex.get("segment_count"),
                        "chunk_source": ex.get("chunk_source"),
                        "unit_of_analysis": ex.get("unit_of_analysis"),
                        "role": ex.get("role"),
                        "actor_label": ex.get("actor_label"),
                        "language": ex.get("language"),
                        "user_type": ex.get("user_type"),
                        "country": ex.get("country"),
                        "has_attachment": ex.get("has_attachment"),
                        "attachment_count": ex.get("attachment_count"),
                        "submission_date": ex.get("submission_date"),
                        "word_count": ex.get("word_count"),
                        "original_text": _truncate_text(ex.get("original_text") or ex.get("text")),
                    }
                    for ex in cluster.get("representative_examples", [])
                ],
            }
        )

    schema_hint = {
        "batch_id": "string",
        "corpus_name": "string",
        "global_assessment": {
            "executive_summary": "string",
            "overall_structure": "string",
            "critical_reading": "string",
            "merger_assessment": "string",
        },
        "interpretations": [
            {
                "cluster_id": "string",
                "frame_label": "string",
                "summary": "string",
                "distinctive_emphasis": "string",
                "overlap_warning": "string",
                "merge_candidate_with": ["string"],
            }
        ],
    }

    instructions = f"""
You will receive a JSON payload containing the full visible result set for a clustering run.
Interpret the clusters comparatively, using only the supplied analysis bundle.

Research question:
{research_question}

Faithfulness protocol:
- Represent each cluster faithfully from the evidence that belongs to that cluster.
- Start from the representative text units, then use the tables only to refine or qualify your reading.
- Treat the contrastive terms as weak lexical hints, not as a substitute for paragraph-level meaning.
- Use metadata lift signals only to qualify a reading that is already visible in the texts.
- Pay attention to `role`: central examples show the cluster core, while variation examples show its edges and possible instability.
- Compare each cluster against the others before naming it.
- If a label could describe all clusters equally well, it is not faithful enough and should be rejected.
- If the clusters mostly share one umbrella discourse, reserve that umbrella framing for the executive summary and global assessment, not for every cluster label.
- If the differences are weak, name the weak difference rather than inventing a strong one.
- Do not add concepts, actors, motives, or policy claims that are not visible in the supplied bundle.

Forbidden moves:
- Do not write horoscope-style summaries that could fit almost any policy corpus.
- Do not justify the existence of a cluster just because the clustering algorithm produced one.
- Do not boil a paragraph-level cluster down to two vague keywords.
- Do not use the same broad label template for multiple clusters unless the correct conclusion is that they are not meaningfully distinct.

What to return:
- One JSON object with keys: batch_id, corpus_name, global_assessment, interpretations.
- Preserve the provided cluster_id values exactly.
- First assess the overall structure critically.
- In global_assessment.executive_summary, write a short top-of-report synthesis of the full result set.
- In global_assessment.overall_structure, summarize what the full clustering result appears to capture.
- In global_assessment.critical_reading, say whether the clusters look genuinely distinct, weakly separated, or partly artifactual.
- In global_assessment.merger_assessment, say whether any clusters look like merge candidates.
- For each cluster, provide a descriptive label, a short summary, what most distinguishes it from the others, and a warning about overlap if needed.
- Use the representative examples as the strongest substantive evidence.
- Use the contrastive terms to sharpen the interpretation, not to replace the text evidence.
- Use the global tables to judge whether the partition may be forced, overlapping, or weak.
- If multiple clusters share one broad discourse, say that and describe only the marginal difference.
- Avoid justifying the existence of a cluster if the evidence for distinctiveness is weak.
- Avoid repeating percentages or counts in the summaries unless they are truly necessary.
- Avoid generic labels that could apply equally well to all clusters.
- Do not add any keys that are not part of the requested structure.
- Keep the output compact so it remains valid JSON.
- Keep "frame_label" descriptive rather than slogan-like.
- Keep "executive_summary" to exactly 150 words and one paragraph.
- Keep "summary" to 2 sentences maximum.
- Keep "distinctive_emphasis" to 1 sentence.
- Keep "overlap_warning" to 1 sentence.
- Use "merge_candidate_with" only when there is a plausible merge candidate; otherwise return an empty array.
- Do not restate the whole table. The report already has the numbers.
- Do not mention embeddings, clustering, Gemini, methods, or prompt instructions.

Expected structure:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

Input payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    return instructions


def build_repair_prompt(
    raw_response: str,
    *,
    corpus_name: str,
    research_question: str,
    issues: Sequence[str] | None = None,
) -> str:
    """Ask Gemini to repair a malformed JSON response."""

    issue_block = ""
    if issues:
        issue_block = "Issues to fix:\n" + "\n".join(f"- {issue}" for issue in issues)

    return f"""
The previous response was not valid JSON.

Research question:
{research_question}

Corpus:
{corpus_name}

Return only valid JSON matching the target structure below.
Do not add any keys beyond the required schema.
Required structure:
{json.dumps({"batch_id": "string", "corpus_name": "string", "global_assessment": {"executive_summary": "string", "overall_structure": "string", "critical_reading": "string", "merger_assessment": "string"}, "interpretations": [{"cluster_id": "string", "frame_label": "string", "summary": "string", "distinctive_emphasis": "string", "overlap_warning": "string", "merge_candidate_with": ["string"]}]}, ensure_ascii=False, indent=2)}

Executive summary requirement:
-  120 to 150 words.
- One paragraph.
- Prefer compact wording over elaborate prose.

{issue_block}

Malformed response to repair:
{_truncate_text(raw_response, limit=2500)}
""".strip()
