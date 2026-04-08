"""Gemini-assisted qualitative interpretation for clustered consultation texts.

This script is designed for the Assignment 2 report pipeline. It ingests a CSV
or JSON file containing cluster representatives, sends batched requests to the
Gemini API, and writes structured interpretations to JSON.

Default behavior:
- Prefer original multilingual text fields.
- Use the official Gemini HTTP endpoint if no client library is available.
- Keep outputs structured and reproducible for downstream report rendering.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt_templates as templates


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


CLUSTER_ID_CANDIDATES = ("cluster_id", "cluster", "cluster_name", "topic_id", "group_id")
TEXT_CANDIDATES = (
    "clean_text",
    "original_text",
    "text",
    "feedback_content_markdown",
    "feedback_content",
    "content",
    "body",
    "passage",
    "excerpt",
)
SOURCE_ID_CANDIDATES = (
    "document_id",
    "response_key",
    "source_id",
    "example_id",
    "response_id",
    "id",
    "row_id",
    "feedback_id",
    "submission_id",
)
RANK_CANDIDATES = ("rank", "score", "distance_to_centroid", "distance", "similarity", "priority")
IGNORED_CONTRASTIVE_TERMS = {"pdf", "ref", "pdf ref"}
SUMMARY_STAT_KEYS = (
    "n_units",
    "pct_of_units",
    "n_parent_responses",
    "pct_of_responses",
    "avg_word_count",
    "pct_with_attachment",
    "pct_attachment_sourced_units",
    "dominant_user_type",
    "dominant_language",
    "dominant_country",
    "dominant_chunk_source",
)
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["batch_id", "corpus_name", "global_assessment", "interpretations"],
    "properties": {
        "batch_id": {"type": "STRING"},
        "corpus_name": {"type": "STRING"},
        "global_assessment": {
            "type": "OBJECT",
            "required": ["executive_summary", "overall_structure", "critical_reading", "merger_assessment"],
            "properties": {
                "executive_summary": {"type": "STRING"},
                "overall_structure": {"type": "STRING"},
                "critical_reading": {"type": "STRING"},
                "merger_assessment": {"type": "STRING"},
            },
        },
        "interpretations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "cluster_id",
                    "frame_label",
                    "summary",
                    "distinctive_emphasis",
                    "overlap_warning",
                    "merge_candidate_with",
                ],
                "properties": {
                    "cluster_id": {"type": "STRING"},
                    "frame_label": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "distinctive_emphasis": {"type": "STRING"},
                    "overlap_warning": {"type": "STRING"},
                    "merge_candidate_with": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
            },
        },
    },
}
REPORTED_CLUSTER_STATS = (
    "n_units",
    "pct_of_units",
    "n_parent_responses",
    "pct_of_responses",
    "avg_word_count",
    "pct_with_attachment",
    "pct_attachment_sourced_units",
    "dominant_user_type",
    "dominant_language",
    "dominant_country",
    "dominant_chunk_source",
)


@dataclass
class Example:
    source_id: str
    original_text: str
    response_key: str = ""
    parent_document_id: int = 0
    segment_index: int = 0
    segment_count: int = 0
    chunk_source: str = ""
    unit_of_analysis: str = ""
    role: str = ""
    actor_label: str = ""
    language: str = ""
    user_type: str = ""
    country: str = ""
    has_attachment: bool = False
    attachment_count: int = 0
    submission_date: str = ""
    word_count: int = 0


@dataclass
class Cluster:
    cluster_id: str
    examples: List[Example] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cluster_size(self) -> int:
        for key in ("n_units", "n_documents", "cluster_size"):
            value = self.metadata.get(key)
            try:
                numeric = int(float(str(value)))
            except Exception:
                continue
            if numeric > 0:
                return numeric
        return len(self.examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret clustered consultation examples with Gemini and write structured JSON."
    )
    parser.add_argument(
        "--config",
        help="Optional YAML config file. If provided, input/output defaults are read from it.",
    )
    parser.add_argument("--input", help="CSV or JSON file with cluster examples.")
    parser.add_argument(
        "--output",
        help="Path to the output JSON file. Defaults to <input_stem>_gemini_interpretations.json next to the input.",
    )
    parser.add_argument(
        "--corpus-name",
        default="Assignment 2 consultation corpus",
        help="Human-readable corpus label used in prompts and output.",
    )
    parser.add_argument(
        "--research-question",
        default=(
            "Which semantic frames recur in the configured compiled corpus, and how are "
            "those frames distributed across actor types, languages, countries, and submission formats?"
        ),
        help="Research question guiding the qualitative interpretation.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL") or "gemini-3.1-flash",
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-3.1-flash.",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "http", "client"),
        default="auto",
        help="API transport. auto tries installed client libraries first, then HTTP.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of clusters to send per request. Use 0 to send all clusters together.",
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=6,
        help="Maximum number of representative examples to keep per cluster.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum tokens requested from Gemini.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Gemini.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for API failures or malformed responses.",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Optional pause in seconds between API calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and batch the input, but do not call Gemini.",
    )
    parser.add_argument(
        "--raw-dir",
        help="Optional directory to write raw Gemini responses for debugging.",
    )
    parser.add_argument(
        "--repair-existing-output",
        action="store_true",
        help="Repair the existing output JSON in place instead of rerunning the full analysis prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    log("Starting Gemini interpretation stage")
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config((project_root / args.config).resolve())

    input_value = args.input or config.get("paths", {}).get("representative_examples")
    output_value = args.output or config.get("paths", {}).get("qualitative_interpretation")
    corpus_name = args.corpus_name
    research_question = args.research_question

    if not input_value:
        raise ValueError("No input file supplied. Pass --input or use --config with representative_examples.")

    input_path = Path(input_value)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = (
        Path(output_value).expanduser().resolve()
        if output_value
        else input_path.with_name(f"{input_path.stem}_gemini_interpretations.json")
    )
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
        log(f"Raw Gemini outputs will be written to: {raw_dir}")

    log(f"Loading cluster representatives from: {input_path}")
    clusters = load_clusters(input_path, examples_per_cluster=args.examples_per_cluster)
    log(f"Loaded {len(clusters)} clusters with up to {args.examples_per_cluster} examples each")
    cluster_summary_value = config.get("paths", {}).get("cluster_summary") if config else None
    if cluster_summary_value:
        cluster_summary_path = Path(cluster_summary_value)
        if not cluster_summary_path.is_absolute():
            cluster_summary_path = (project_root / cluster_summary_path).resolve()
        if cluster_summary_path.exists():
            log("Enriching clusters with summary table")
            enrich_clusters_with_summary(clusters, load_cluster_summary_lookup(cluster_summary_path))
    cluster_terms_value = config.get("paths", {}).get("cluster_terms") if config else None
    if cluster_terms_value:
        cluster_terms_path = Path(cluster_terms_value)
        if not cluster_terms_path.is_absolute():
            cluster_terms_path = (project_root / cluster_terms_path).resolve()
        if cluster_terms_path.exists():
            log("Enriching clusters with top-term table")
            enrich_clusters_with_terms(clusters, load_cluster_terms_lookup(cluster_terms_path))
    metadata_signals_value = config.get("paths", {}).get("cluster_metadata_signals") if config else None
    if metadata_signals_value:
        metadata_signals_path = Path(metadata_signals_value)
        if not metadata_signals_path.is_absolute():
            metadata_signals_path = (project_root / metadata_signals_path).resolve()
        if metadata_signals_path.exists():
            log("Enriching clusters with metadata signals")
            enrich_clusters_with_metadata_signals(clusters, load_metadata_signal_lookup(metadata_signals_path))
    log("Building the full analysis bundle for Gemini")
    analysis_bundle = build_analysis_bundle(config, project_root, clusters)
    log(f"Analysis bundle tables available: {', '.join(sorted(analysis_bundle.keys()))}")
    batches = make_batches(clusters, batch_size=args.batch_size)
    log(f"Prepared {len(batches)} Gemini batch(es)")

    result = {
        "project": "Assignment 2 qualitative interpretation",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "corpus_name": corpus_name,
        "research_question": research_question,
        "model": args.model,
        "transport": args.transport,
        "prompt_contract": {
            "cluster_stats_fields": list(REPORTED_CLUSTER_STATS),
            "top_terms_included": True,
            "global_tables_included": sorted(analysis_bundle.keys()),
            "representative_examples_per_cluster": args.examples_per_cluster,
            "joint_review": True,
            "global_assessment_mode": "single_pass" if len(batches) == 1 else "separate_full_corpus_pass",
            "output_fields": [
                "global_assessment",
                "cluster_id",
                "frame_label",
                "summary",
                "distinctive_emphasis",
                "overlap_warning",
                "merge_candidate_with",
            ],
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cluster_count": len(clusters),
        "batch_count": len(batches),
        "batches": [],
        "global_assessment": {},
        "interpretations": [],
    }

    if args.dry_run:
        log("Dry run requested; writing batch plan only")
        for i, batch in enumerate(batches, start=1):
            result["batches"].append(
                {
                    "batch_id": f"batch-{i}",
                    "cluster_ids": [c.cluster_id for c in batch],
                    "cluster_sizes": {c.cluster_id: c.cluster_size for c in batch},
                }
            )
        write_json(output_path, result)
        return 0

    if not args.api_key:
        raise ValueError(
            "No API key supplied. Set GEMINI_API_KEY or pass --api-key."
        )

    client_mode = resolve_transport(args.transport)
    log(f"Using Gemini transport: {client_mode}")

    if args.repair_existing_output:
        log("Repairing existing Gemini output in place")
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot repair missing output file: {output_path}")
        current_output = load_json(output_path)
        if not isinstance(current_output, Mapping):
            raise ValueError("Existing output is not a JSON object and cannot be repaired.")
        expected_cluster_ids = [c.cluster_id for c in clusters]
        issues = validate_gemini_payload(current_output, expected_cluster_ids=expected_cluster_ids)
        if not issues:
            return 0
        repair_prompt = templates.build_repair_prompt(
            json.dumps(current_output, ensure_ascii=False, indent=2),
            corpus_name=corpus_name,
            research_question=research_question,
            issues=issues,
        )
        raw_text = call_gemini(
            prompt=repair_prompt,
            api_key=args.api_key,
            model=args.model,
            transport=client_mode,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
        repaired = parse_json_response(raw_text)
        issues = validate_gemini_payload(repaired, expected_cluster_ids=expected_cluster_ids)
        if issues:
            raise RuntimeError(f"Repair output is still invalid: {'; '.join(issues)}")

        updated = dict(current_output)
        updated["model"] = args.model
        updated["transport"] = args.transport
        updated["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        updated["global_assessment"] = normalize_global_assessment(repaired.get("global_assessment"))
        updated["interpretations"] = [
            normalize_interpretation(item, batch_id="repair-batch")
            for item in repaired.get("interpretations", [])
        ]
        write_json(output_path, updated)
        return 0

    for batch_index, batch in enumerate(batches, start=1):
        batch_id = f"batch-{batch_index}"
        log(
            f"Gemini batch {batch_index}/{len(batches)}: "
            f"clusters={[cluster.cluster_id for cluster in batch]}"
        )
        prompt = templates.build_batch_prompt(
            batch_id,
            [cluster_to_prompt_payload(cluster) for cluster in batch],
            analysis_bundle=analysis_bundle,
            corpus_name=corpus_name,
            research_question=research_question,
        )

        parsed, _raw_text = request_gemini_payload(
            batch_id=batch_id,
            prompt=prompt,
            validator=lambda payload, expected_cluster_ids=[c.cluster_id for c in batch]: validate_gemini_payload(
                payload,
                expected_cluster_ids=expected_cluster_ids,
            ),
            api_key=args.api_key,
            model=args.model,
            transport=client_mode,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            retries=args.retries,
            raw_dir=raw_dir,
            corpus_name=corpus_name,
            research_question=research_question,
        )

        interpretations = parsed.get("interpretations")
        if not isinstance(interpretations, list):
            raise ValueError(f"Gemini response for {batch_id} missing interpretations list.")
        if len(batches) == 1 and "global_assessment" in parsed and not result["global_assessment"]:
            result["global_assessment"] = normalize_global_assessment(parsed.get("global_assessment"))

        result["batches"].append(
            {
                "batch_id": batch_id,
                "cluster_ids": [c.cluster_id for c in batch],
                "cluster_sizes": {c.cluster_id: c.cluster_size for c in batch},
                "response_keys": sorted(parsed.keys()),
            }
        )
        for item in interpretations:
            result["interpretations"].append(normalize_interpretation(item, batch_id=batch_id))
        log(f"Gemini batch {batch_id} completed")

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    if len(batches) > 1 or not result["global_assessment"]:
        global_batch_id = "global-assessment"
        log(
            f"Gemini {global_batch_id}: synthesizing corpus-wide assessment "
            f"across {len(clusters)} clusters"
        )
        global_prompt = templates.build_global_assessment_prompt(
            global_batch_id,
            [cluster_to_prompt_payload(cluster, max_examples=2) for cluster in clusters],
            analysis_bundle=analysis_bundle,
            cluster_interpretations=result["interpretations"],
            corpus_name=corpus_name,
            research_question=research_question,
        )
        global_payload, _global_raw_text = request_gemini_payload(
            batch_id=global_batch_id,
            prompt=global_prompt,
            validator=lambda payload: validate_global_assessment_payload(payload, cluster_count=len(clusters)),
            api_key=args.api_key,
            model=args.model,
            transport=client_mode,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            retries=args.retries,
            raw_dir=raw_dir,
            corpus_name=corpus_name,
            research_question=research_question,
        )
        result["global_assessment"] = normalize_global_assessment(global_payload.get("global_assessment"))
        result["batches"].append(
            {
                "batch_id": global_batch_id,
                "cluster_ids": [c.cluster_id for c in clusters],
                "cluster_sizes": {c.cluster_id: c.cluster_size for c in clusters},
                "response_keys": sorted(global_payload.keys()),
                "mode": "full_corpus_global_assessment",
            }
        )
        log(f"Gemini {global_batch_id} completed")

    result["batch_count"] = len(result["batches"])
    write_json(output_path, result)
    log(f"Gemini interpretation written to: {output_path}")
    return 0


def request_gemini_payload(
    *,
    batch_id: str,
    prompt: str,
    validator: Callable[[Mapping[str, Any]], List[str]],
    api_key: str,
    model: str,
    transport: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
    retries: int,
    raw_dir: Optional[Path],
    corpus_name: str,
    research_question: str,
) -> Tuple[Dict[str, Any], str]:
    raw_text = ""
    parsed = None
    last_error: Optional[Exception] = None
    next_max_output_tokens = max(max_output_tokens, 4096)
    for attempt in range(1, max(1, retries) + 1):
        attempt_max_output_tokens = next_max_output_tokens
        try:
            raw_text = call_gemini(
                prompt=prompt,
                api_key=api_key,
                model=model,
                transport=transport,
                timeout=timeout,
                max_output_tokens=attempt_max_output_tokens,
                temperature=temperature,
            )
            try:
                parsed = parse_json_response(raw_text)
            except Exception:
                repair_prompt = templates.build_repair_prompt(
                    raw_text,
                    corpus_name=corpus_name,
                    research_question=research_question,
                    issues=["Previous response was malformed or truncated JSON. Return a complete valid JSON object."],
                )
                raw_text = call_gemini(
                    prompt=repair_prompt,
                    api_key=api_key,
                    model=model,
                    transport=transport,
                    timeout=timeout,
                    max_output_tokens=max(attempt_max_output_tokens, 7000),
                    temperature=0.0,
                )
                parsed = parse_json_response(raw_text)
            issues = validator(parsed)
            if issues:
                repair_prompt = templates.build_repair_prompt(
                    raw_text,
                    corpus_name=corpus_name,
                    research_question=research_question,
                    issues=issues,
                )
                raw_text = call_gemini(
                    prompt=repair_prompt,
                    api_key=api_key,
                    model=model,
                    transport=transport,
                    timeout=timeout,
                    max_output_tokens=max(attempt_max_output_tokens, 7000),
                    temperature=0.0,
                )
                parsed = parse_json_response(raw_text)
                issues = validator(parsed)
                if issues:
                    raise ValueError("; ".join(issues))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            parsed = None
            if raw_dir:
                write_json(
                    raw_dir / f"{batch_id}_attempt_{attempt}_failed.json",
                    {
                        "prompt": prompt,
                        "response_text": raw_text,
                        "error": str(exc),
                        "max_output_tokens": attempt_max_output_tokens,
                    },
                )
            if attempt >= retries:
                break
            next_max_output_tokens = min(
                max(attempt_max_output_tokens + 1500, int(attempt_max_output_tokens * 1.75)),
                12000,
            )
            log(
                f"Gemini batch {batch_id} attempt {attempt} failed; "
                f"retrying with max_output_tokens={next_max_output_tokens}. "
                f"Reason: {exc}"
            )
            time.sleep(min(2**attempt, 8))
    if parsed is None:
        if last_error is None:
            raise RuntimeError(f"Failed to obtain a response for {batch_id}")
        raise RuntimeError(f"Failed to parse Gemini response for {batch_id}: {last_error}") from last_error

    if raw_dir:
        write_json(raw_dir / f"{batch_id}_raw.json", {"prompt": prompt, "response_text": raw_text})
    return parsed, raw_text


def resolve_transport(requested: str) -> str:
    if requested in {"http", "client"}:
        return requested
    if can_use_google_genai():
        return "client"
    if can_use_google_generativelanguage():
        return "client"
    return "http"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def can_use_google_genai() -> bool:
    try:
        import google.genai  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def can_use_google_generativelanguage() -> bool:
    try:
        import google.generativeai  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def load_clusters(path: Path, *, examples_per_cluster: int) -> List[Cluster]:
    if path.suffix.lower() == ".csv":
        rows = load_csv_rows(path)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[extract_cluster_id(row)].append(row)
        clusters = [build_cluster(cluster_id, items, examples_per_cluster) for cluster_id, items in grouped.items()]
        return sorted(clusters, key=lambda c: natural_key(c.cluster_id))
    if path.suffix.lower() in {".json", ".jsonl"}:
        payload = load_json(path)
        if isinstance(payload, list):
            clusters = []
            if payload and all(isinstance(item, Mapping) and "examples" in item for item in payload):
                for item in payload:
                    cluster_id = extract_cluster_id_from_mapping(item, default=str(len(clusters) + 1))
                    examples = item.get("examples", [])
                    clusters.append(build_cluster(cluster_id, list(examples), examples_per_cluster, item))
            elif payload and all(isinstance(item, Mapping) and extract_cluster_id(item) for item in payload):
                grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for item in payload:
                    grouped[extract_cluster_id(item)].append(dict(item))
                clusters = [build_cluster(cluster_id, items, examples_per_cluster) for cluster_id, items in grouped.items()]
            else:
                clusters = [build_cluster("cluster-1", payload, examples_per_cluster)]
            return sorted(clusters, key=lambda c: natural_key(c.cluster_id))
        if isinstance(payload, Mapping):
            if "clusters" in payload and isinstance(payload["clusters"], list):
                return load_clusters_from_cluster_list(payload["clusters"], examples_per_cluster)
            if all(isinstance(v, list) for v in payload.values()):
                clusters = []
                for cluster_id, examples in payload.items():
                    clusters.append(build_cluster(str(cluster_id), list(examples), examples_per_cluster))
                return sorted(clusters, key=lambda c: natural_key(c.cluster_id))
            return [build_cluster("cluster-1", [dict(payload)], examples_per_cluster)]
    raise ValueError(f"Unsupported input format: {path.suffix}")


def extract_cluster_id(row: Mapping[str, Any]) -> str:
    cluster_id = first_nonempty(row, CLUSTER_ID_CANDIDATES)
    return cluster_id.strip() if cluster_id else ""


def load_clusters_from_cluster_list(items: Sequence[Any], examples_per_cluster: int) -> List[Cluster]:
    clusters: List[Cluster] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        cluster_id = extract_cluster_id_from_mapping(item, default=str(len(clusters) + 1))
        examples = item.get("examples", [])
        clusters.append(build_cluster(cluster_id, list(examples), examples_per_cluster, item))
    return sorted(clusters, key=lambda c: natural_key(c.cluster_id))


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cluster_summary_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        lookup: Dict[str, Dict[str, Any]] = {}
        for row in reader:
            cluster_id = str(row.get("cluster_id") or "").strip()
            if cluster_id:
                lookup[cluster_id] = dict(row)
        return lookup


def enrich_clusters_with_summary(clusters: Sequence[Cluster], lookup: Mapping[str, Mapping[str, Any]]) -> None:
    for cluster in clusters:
        summary_row = lookup.get(str(cluster.cluster_id))
        if not summary_row:
            continue
        merged = dict(summary_row)
        merged.update(cluster.metadata)
        cluster.metadata = merged


def load_cluster_terms_lookup(path: Path, *, max_terms: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    """Load the strongest contrastive terms for each cluster.

    The report already displays a term table built from the same file. Feeding the
    same terms into Gemini keeps the qualitative layer aligned with the visible
    evidence instead of asking the model to infer labels from examples alone.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in reader:
            cluster_id = str(row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            grouped[cluster_id].append(
                {
                    "term": str(row.get("term") or "").strip(),
                    "term_rank": coerce_int(row.get("term_rank")),
                    "contrast_score": coerce_float(row.get("contrast_score")),
                    "salience_score": coerce_float(row.get("salience_score")),
                }
            )

    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for cluster_id, items in grouped.items():
        items = [item for item in items if should_keep_term(item.get("term"))]
        items = sorted(
            items,
            key=lambda item: (
                item.get("term_rank", 999999),
                -float(item.get("salience_score") or 0.0),
            ),
        )
        lookup[cluster_id] = items[:max_terms]
    return lookup


def enrich_clusters_with_terms(
    clusters: Sequence[Cluster], terms_lookup: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    for cluster in clusters:
        cluster.metadata["top_terms"] = list(terms_lookup.get(str(cluster.cluster_id), []))


def load_metadata_signal_lookup(path: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        return {}
    output: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for field, cluster_payload in payload.items():
        if not isinstance(cluster_payload, Mapping):
            continue
        for cluster_id, rows in cluster_payload.items():
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                output[str(cluster_id)][str(field)] = [dict(item) for item in rows if isinstance(item, Mapping)]
    return dict(output)


def enrich_clusters_with_metadata_signals(
    clusters: Sequence[Cluster],
    signals_lookup: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    for cluster in clusters:
        cluster.metadata["metadata_signals"] = {
            field: list(rows)
            for field, rows in signals_lookup.get(str(cluster.cluster_id), {}).items()
        }


def build_analysis_bundle(
    config: Mapping[str, Any], project_root: Path, clusters: Sequence[Cluster]
) -> Dict[str, Any]:
    """Assemble the full visible result set for Gemini.

    The user asked that Gemini should see the report dataframes together rather
    than only isolated cluster snippets. This bundle mirrors the main tables we
    can currently surface in the report: corpus snapshot, model-selection
    metrics, overlap across clusters, validation, cluster statistics, top terms,
    and representative evidence.
    """
    if not config:
        return {}

    paths = config.get("paths", {})
    bundle: Dict[str, Any] = {}

    metadata_summary = load_optional_json(project_root, paths.get("metadata_summary"))
    if isinstance(metadata_summary, Mapping):
        chunk_counts = metadata_summary.get("chunk_source_counts") or {}
        bundle["corpus_snapshot"] = {
            "raw_documents": metadata_summary.get("n_raw_documents"),
            "retained_responses": metadata_summary.get("n_retained_responses"),
            "analysis_units": metadata_summary.get("n_units"),
            "unit_of_analysis": metadata_summary.get("unit_of_analysis"),
            "avg_response_word_count": metadata_summary.get("avg_response_word_count"),
            "avg_unit_word_count": metadata_summary.get("avg_unit_word_count"),
            "avg_units_per_response": metadata_summary.get("avg_units_per_response"),
            "attachment_rate": metadata_summary.get("attachment_rate"),
            "chunk_source_counts": chunk_counts,
        }

    model_selection_rows = load_optional_csv(project_root, paths.get("model_selection"))
    if model_selection_rows:
        bundle["model_selection_metrics"] = [
            {
                "cluster_count": coerce_int(row.get("cluster_count")),
                "silhouette_score": round(coerce_float(row.get("silhouette_score")), 6),
                "davies_bouldin_score": round(coerce_float(row.get("davies_bouldin_score")), 6),
                "calinski_harabasz_score": round(coerce_float(row.get("calinski_harabasz_score")), 6),
                "inertia": round(coerce_float(row.get("inertia")), 4),
            }
            for row in model_selection_rows
        ]

    clustering_decision = load_optional_json(project_root, paths.get("clustering_decision"))
    if isinstance(clustering_decision, Mapping):
        bundle["clustering_decision"] = {
            "selection_metric": clustering_decision.get("selection_metric"),
            "selected_k": clustering_decision.get("selected_k"),
            "best_row": clustering_decision.get("best_row"),
        }

    cluster_summary_rows = load_optional_csv(project_root, paths.get("cluster_summary"))
    if cluster_summary_rows:
        bundle["cluster_statistics"] = [
            {
                "cluster_id": str(row.get("cluster_id") or "").strip(),
                "n_units": coerce_int(row.get("n_units") or row.get("n_documents")),
                "pct_of_units": round(coerce_float(row.get("pct_of_units") or row.get("pct_of_corpus")), 4),
                "n_parent_responses": coerce_int(row.get("n_parent_responses") or row.get("n_documents")),
                "pct_of_responses": round(
                    coerce_float(row.get("pct_of_responses") or row.get("pct_of_corpus")), 4
                ),
                "avg_word_count": round(coerce_float(row.get("avg_word_count")), 2),
                "pct_attachment_sourced_units": round(coerce_float(row.get("pct_attachment_sourced_units")), 4),
                "dominant_user_type": str(row.get("dominant_user_type") or "").strip(),
                "dominant_language": str(row.get("dominant_language") or "").strip(),
                "dominant_country": str(row.get("dominant_country") or "").strip(),
                "medoid_text_preview": str(row.get("medoid_text_preview") or "").strip(),
            }
            for row in cluster_summary_rows
        ]

    terms_rows = load_optional_csv(project_root, paths.get("cluster_terms"))
    if terms_rows:
        grouped_terms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in terms_rows:
            term = str(row.get("term") or "").strip()
            if not should_keep_term(term):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            grouped_terms[cluster_id].append(
                {
                    "term_rank": coerce_int(row.get("term_rank")),
                    "term": term,
                    "contrast_score": round(coerce_float(row.get("contrast_score")), 4),
                    "salience_score": round(coerce_float(row.get("salience_score")), 6),
                }
            )
        bundle["cluster_terms_table"] = {
            cluster_id: sorted(items, key=lambda item: item["term_rank"])[:12]
            for cluster_id, items in grouped_terms.items()
        }

    metadata_signal_payload = load_optional_json(project_root, paths.get("cluster_metadata_signals"))
    if isinstance(metadata_signal_payload, Mapping):
        bundle["cluster_metadata_signals"] = metadata_signal_payload

    representative_rows = load_optional_csv(project_root, paths.get("representative_examples_flat"))
    if representative_rows:
        grouped_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in representative_rows:
            cluster_id = str(row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            grouped_examples[cluster_id].append(row)
        table_rows: List[Dict[str, Any]] = []
        for cluster_id, items in grouped_examples.items():
            items = sorted(items, key=lambda row: coerce_float(row.get("distance_to_centroid")))
            for row in items[:3]:
                table_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "role": str(row.get("role") or "").strip(),
                        "actor_label": str(row.get("actor_label") or "").strip(),
                        "user_type": str(row.get("user_type") or "").strip(),
                        "language": str(row.get("language") or "").strip(),
                        "chunk_source": str(row.get("chunk_source") or "").strip(),
                        "distance_to_centroid": round(coerce_float(row.get("distance_to_centroid")), 6),
                        "clean_text": truncate_for_bundle(row.get("clean_text"), limit=320),
                    }
                )
        bundle["representative_evidence_table"] = table_rows

    centroid_similarity_rows = load_optional_csv(project_root, paths.get("centroid_similarity"))
    if centroid_similarity_rows:
        bundle["centroid_similarity"] = [
            {
                "cluster_left": coerce_int(row.get("cluster_left")),
                "cluster_right": coerce_int(row.get("cluster_right")),
                "cosine_similarity": round(coerce_float(row.get("cosine_similarity")), 6),
            }
            for row in centroid_similarity_rows
        ]

    prepared_rows = load_optional_csv(project_root, paths.get("prepared_corpus"))
    if prepared_rows:
        actor_counts: Dict[str, set[str]] = defaultdict(set)
        language_counts: Dict[str, set[str]] = defaultdict(set)
        for row in prepared_rows:
            parent_document_id = str(row.get("parent_document_id") or "").strip()
            if not parent_document_id:
                continue
            user_type = str(row.get("user_type") or "").strip()
            language = str(row.get("language") or "").strip()
            if user_type:
                actor_counts[user_type].add(parent_document_id)
            if language:
                language_counts[language].add(parent_document_id)
        bundle["top_actor_categories"] = summarize_membership_counts(actor_counts, top_n=10)
        bundle["top_languages"] = summarize_membership_counts(language_counts, top_n=10)

    cluster_assignments_value = paths.get("cluster_assignments")
    if cluster_assignments_value:
        cluster_assignments_path = Path(cluster_assignments_value)
        if not cluster_assignments_path.is_absolute():
            cluster_assignments_path = (project_root / cluster_assignments_path).resolve()
        if cluster_assignments_path.exists():
            overlap_rows = build_response_frame_distribution(cluster_assignments_path)
            if overlap_rows:
                bundle["response_frame_overlap"] = overlap_rows

    validation_summary = load_optional_json(project_root, paths.get("cluster_validation_summary"))
    if isinstance(validation_summary, Mapping):
        null_summary = validation_summary.get("null_summary") or {}
        observed_metrics = validation_summary.get("observed_metrics") or {}
        bundle["validation_table"] = {
            "observed_cluster_count": observed_metrics.get("cluster_count"),
            "observed_silhouette_score": observed_metrics.get("silhouette_score"),
            "permutation_p_value": null_summary.get("permutation_p_value"),
        }

    bundle["cluster_ids_in_run"] = [cluster.cluster_id for cluster in clusters]
    return bundle


def load_optional_json(project_root: Path, value: Any) -> Any:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.exists():
        return None
    return load_json(path)


def load_optional_csv(project_root: Path, value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.exists():
        return []
    return load_csv_rows(path)


def build_response_frame_distribution(path: Path) -> List[Dict[str, Any]]:
    response_to_clusters: Dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parent_document_id = str(row.get("parent_document_id") or "").strip()
            cluster_id = str(row.get("cluster_id") or "").strip()
            if parent_document_id and cluster_id:
                response_to_clusters[parent_document_id].add(cluster_id)

    frame_counts: Dict[int, int] = defaultdict(int)
    total = 0
    for cluster_ids in response_to_clusters.values():
        frame_counts[len(cluster_ids)] += 1
        total += 1

    records = []
    for n_frames in sorted(frame_counts):
        n_responses = frame_counts[n_frames]
        pct = round(100.0 * n_responses / total, 4) if total else 0.0
        records.append({"n_frames": n_frames, "n_responses": n_responses, "pct": pct})
    return records


def truncate_for_bundle(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def summarize_membership_counts(groups: Mapping[str, set[str]], top_n: int) -> List[Dict[str, Any]]:
    total = sum(len(members) for members in groups.values())
    rows: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        count = len(members)
        pct = round(100.0 * count / total, 4) if total else 0.0
        rows.append({"category": key, "n": count, "pct": pct})
    return rows[:top_n]


def extract_cluster_id_from_mapping(item: Mapping[str, Any], default: str) -> str:
    for key in ("cluster_id", "cluster", "id"):
        if key not in item:
            continue
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return default


def build_cluster(
    cluster_id: str,
    rows: Sequence[Mapping[str, Any]],
    examples_per_cluster: int,
    cluster_payload: Optional[Mapping[str, Any]] = None,
) -> Cluster:
    normalized_rows = [normalize_row(row) for row in rows]
    normalized_rows = sorted(normalized_rows, key=sort_key_for_row)
    examples = [row_to_example(row) for row in normalized_rows[: max(1, examples_per_cluster)]]
    metadata = {}
    if cluster_payload:
        metadata = {
            key: value
            for key, value in cluster_payload.items()
            if key not in {"cluster_id", "cluster", "id", "examples"}
        }
    if not metadata:
        metadata = infer_cluster_metadata(normalized_rows)
    return Cluster(cluster_id=str(cluster_id), examples=examples, metadata=metadata)


def normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in row.items()}


def row_to_example(row: Mapping[str, Any]) -> Example:
    source_id = first_nonempty(row, SOURCE_ID_CANDIDATES) or ""
    original_text = first_nonempty(row, TEXT_CANDIDATES) or ""
    actor_label = str(row.get("actor_label") or row.get("actor") or row.get("organization") or "").strip()
    language = str(row.get("language") or row.get("lang") or "").strip()
    user_type = str(row.get("user_type") or row.get("actor_type") or row.get("type") or "").strip()
    country = str(row.get("country") or row.get("country_code") or "").strip()
    return Example(
        source_id=str(source_id),
        original_text=str(original_text),
        response_key=str(row.get("response_key") or "").strip(),
        parent_document_id=coerce_int(row.get("parent_document_id")),
        segment_index=coerce_int(row.get("segment_index")),
        segment_count=coerce_int(row.get("segment_count")),
        chunk_source=str(row.get("chunk_source") or "").strip(),
        unit_of_analysis=str(row.get("unit_of_analysis") or "").strip(),
        role=str(row.get("role") or "").strip(),
        actor_label=actor_label,
        language=language,
        user_type=user_type,
        country=country,
        has_attachment=coerce_bool(row.get("has_attachment")),
        attachment_count=coerce_int(row.get("attachment_count")),
        submission_date=str(row.get("submission_date") or "").strip(),
        word_count=coerce_int(row.get("word_count")),
    )


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def coerce_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def coerce_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def should_keep_term(value: Any) -> bool:
    text = " ".join(str(value or "").strip().lower().split())
    if text == "":
        return False
    return text not in IGNORED_CONTRASTIVE_TERMS


def infer_cluster_metadata(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for field in ("language", "user_type", "country"):
        values = [str(row.get(field) or "").strip() for row in rows if str(row.get(field) or "").strip()]
        if values:
            metadata[field] = top_values(values)
    if rows:
        metadata["cluster_size"] = len(rows)
    return metadata


def top_values(values: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def first_nonempty(row: Mapping[str, Any], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def sort_key_for_row(row: Mapping[str, Any]) -> Tuple[int, float, str]:
    for field in RANK_CANDIDATES:
        value = row.get(field)
        if value is None:
            continue
        try:
            numeric = float(str(value).strip())
        except Exception:
            continue
        if field in {"rank", "priority"}:
            return (0, numeric, "")
        if field == "score":
            return (0, -numeric, "")
        if field == "distance":
            return (0, numeric, "")
        if field == "similarity":
            return (0, -numeric, "")
    source_id = first_nonempty(row, SOURCE_ID_CANDIDATES)
    return (1, 0.0, source_id)


def make_batches(clusters: Sequence[Cluster], *, batch_size: int) -> List[List[Cluster]]:
    if batch_size <= 0 or batch_size >= len(clusters):
        return [list(clusters)]
    batches: List[List[Cluster]] = []
    for start in range(0, len(clusters), batch_size):
        batches.append(list(clusters[start : start + batch_size]))
    return batches


def cluster_to_prompt_payload(cluster: Cluster, *, max_examples: Optional[int] = None) -> Dict[str, Any]:
    # Gemini only gets the analysis bundle that the report can defend:
    # 1) cluster-level stats already shown in tables,
    # 2) the strongest contrastive terms from the inspection layer, and
    # 3) representative evidence units closest to the cluster centroid.
    representative_examples = cluster.examples
    if max_examples is not None:
        representative_examples = representative_examples[: max(0, max_examples)]
    return {
        "cluster_id": cluster.cluster_id,
        "cluster_statistics": build_cluster_stats_payload(cluster),
        "contrastive_terms": build_top_terms_payload(cluster),
        "metadata_signals": cluster.metadata.get("metadata_signals", []),
        "representative_examples": [
            {
                "source_id": ex.source_id,
                "response_key": ex.response_key,
                "parent_document_id": ex.parent_document_id,
                "segment_index": ex.segment_index,
                "segment_count": ex.segment_count,
                "chunk_source": ex.chunk_source,
                "unit_of_analysis": ex.unit_of_analysis,
                "role": ex.role,
                "actor_label": ex.actor_label,
                "language": ex.language,
                "user_type": ex.user_type,
                "country": ex.country,
                "has_attachment": ex.has_attachment,
                "attachment_count": ex.attachment_count,
                "submission_date": ex.submission_date,
                "word_count": ex.word_count,
                "original_text": ex.original_text,
            }
            for ex in representative_examples
        ],
    }


def build_cluster_stats_payload(cluster: Cluster) -> Dict[str, Any]:
    """Expose only the cluster statistics that matter in the report.

    The source CSV can contain a lot of extra bookkeeping fields. Restricting the
    prompt to the visible report metrics makes the qualitative layer easier to
    inspect and less likely to latch onto accidental columns.
    """
    stats: Dict[str, Any] = {"cluster_size": cluster.cluster_size}
    for key in REPORTED_CLUSTER_STATS:
        if key in cluster.metadata:
            stats[key] = normalize_scalar(cluster.metadata.get(key))
    return stats


def build_top_terms_payload(cluster: Cluster) -> List[Dict[str, Any]]:
    terms = cluster.metadata.get("top_terms") or []
    result: List[Dict[str, Any]] = []
    for item in terms:
        if not isinstance(item, Mapping):
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        result.append(
            {
                "term": term,
                "term_rank": coerce_int(item.get("term_rank")),
                "contrast_score": round(coerce_float(item.get("contrast_score")), 4),
                "salience_score": round(coerce_float(item.get("salience_score")), 6),
            }
        )
    return result


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
    except Exception:
        return text
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 4)


def call_gemini(
    *,
    prompt: str,
    api_key: str,
    model: str,
    transport: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
) -> str:
    prompt = f"{templates.SYSTEM_PROMPT}\n\n{prompt}"
    if transport == "client":
        client_text = call_gemini_via_client(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if client_text:
            return client_text
    return call_gemini_via_http(
        prompt=prompt,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def call_gemini_via_http(
    *,
    prompt: str,
    api_key: str,
    model: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = build_http_payload(prompt, max_output_tokens, temperature, include_schema=True)
    try:
        data = post_http_payload(url, payload, timeout=timeout)
    except RuntimeError as exc:
        message = str(exc)
        if "responseSchema" in message or "response_schema" in message or "Unknown name" in message:
            fallback_payload = build_http_payload(prompt, max_output_tokens, temperature, include_schema=False)
            data = post_http_payload(url, fallback_payload, timeout=timeout)
        else:
            raise
    return extract_response_text(data)


def call_gemini_via_client(
    *,
    prompt: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
            },
        )
        text = getattr(response, "text", None)
        if text:
            return str(text)
        return extract_client_text(response)
    except Exception:
        pass

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
            },
        )
        text = getattr(response, "text", None)
        if text:
            return str(text)
        return extract_client_text(response)
    except Exception:
        return ""


def extract_client_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    candidates = []
    if hasattr(response, "candidates"):
        candidates = getattr(response, "candidates") or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if parts:
            chunks = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))
            if chunks:
                return "\n".join(chunks)
    return ""


def build_http_payload(
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    *,
    include_schema: bool,
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    if include_schema:
        generation_config["responseSchema"] = GEMINI_RESPONSE_SCHEMA
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }


def post_http_payload(url: str, payload: Mapping[str, Any], *, timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP error {exc.code}: {body}") from exc


def extract_response_text(data: Mapping[str, Any]) -> str:
    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, Mapping)]
        text = "\n".join(t for t in texts if t)
        if text:
            return text
    raise ValueError(f"Gemini response did not contain text: {data}")


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty Gemini response.")
    for candidate in (raw_text, extract_json_block(raw_text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Gemini response was not valid JSON: {raw_text[:500]}")


def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return ""


def normalize_interpretation(item: Any, *, batch_id: str) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"batch_id": batch_id, "raw_item": item}
    return {
        "batch_id": batch_id,
        "cluster_id": str(item.get("cluster_id", "")),
        "frame_label": str(item.get("frame_label", "")).strip(),
        "summary": str(item.get("summary", "")).strip(),
        "distinctive_emphasis": str(item.get("distinctive_emphasis", "")).strip(),
        "overlap_warning": str(item.get("overlap_warning", "")).strip(),
        "merge_candidate_with": normalize_string_list(item.get("merge_candidate_with")),
    }


def normalize_global_assessment(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "executive_summary": str(value.get("executive_summary", "")).strip(),
        "overall_structure": str(value.get("overall_structure", "")).strip(),
        "critical_reading": str(value.get("critical_reading", "")).strip(),
        "merger_assessment": str(value.get("merger_assessment", "")).strip(),
    }


def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return [str(value)]
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def validate_gemini_payload(payload: Mapping[str, Any], expected_cluster_ids: Sequence[str]) -> List[str]:
    issues: List[str] = []

    global_assessment = payload.get("global_assessment")
    if not isinstance(global_assessment, Mapping):
        issues.append("Missing global_assessment object.")
    else:
        executive_summary = str(global_assessment.get("executive_summary", "")).strip()
        word_count = count_words(executive_summary)
        if word_count < 80:
            issues.append(f"Executive summary is too short to be useful ({word_count} words).")
        elif word_count > 220:
            issues.append(f"Executive summary is too long and likely drifting ({word_count} words).")

    interpretations = payload.get("interpretations")
    if not isinstance(interpretations, Sequence) or isinstance(interpretations, str):
        issues.append("interpretations must be a list.")
        return issues

    seen_cluster_ids = {
        str(item.get("cluster_id", "")).strip()
        for item in interpretations
        if isinstance(item, Mapping)
    }
    expected = {str(cluster_id).strip() for cluster_id in expected_cluster_ids}
    if seen_cluster_ids != expected:
        issues.append(
            f"interpretations must cover exactly these cluster_ids: {sorted(expected)}; got {sorted(seen_cluster_ids)}."
        )
    return issues


def validate_global_assessment_payload(payload: Mapping[str, Any], *, cluster_count: int) -> List[str]:
    issues: List[str] = []

    global_assessment = payload.get("global_assessment")
    if not isinstance(global_assessment, Mapping):
        issues.append("Missing global_assessment object.")
    else:
        executive_summary = str(global_assessment.get("executive_summary", "")).strip()
        word_count = count_words(executive_summary)
        if word_count < 80:
            issues.append(f"Executive summary is too short to be useful ({word_count} words).")
        elif word_count > 220:
            issues.append(f"Executive summary is too long and likely drifting ({word_count} words).")

        combined_text = " ".join(
            str(global_assessment.get(key, "")).strip().lower()
            for key in ("executive_summary", "overall_structure", "critical_reading", "merger_assessment")
        )
        if cluster_count > 1 and looks_like_single_cluster_claim(combined_text):
            issues.append(
                f"Global assessment incorrectly describes the run as one cluster although the run contains {cluster_count} clusters."
            )

    interpretations = payload.get("interpretations")
    if not isinstance(interpretations, Sequence) or isinstance(interpretations, str):
        issues.append("interpretations must be a list.")
    elif len(list(interpretations)) != 0:
        issues.append("interpretations must be an empty list for the global-assessment pass.")

    return issues


def looks_like_single_cluster_claim(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    normalized = re.sub(r"\bnot (?:just )?(?:a )?single cluster\b", "", normalized)
    normalized = re.sub(r"\bnot (?:just )?(?:one|only one) cluster\b", "", normalized)
    patterns = (
        r"\bthere (?:is|appears to be|seems to be) (?:only )?one cluster\b",
        r"\b(?:this|the run|the result|it) (?:is|looks|appears|seems) (?:effectively |basically )?(?:a )?single cluster\b",
        r"\bone cluster only\b",
        r"\bonly one cluster\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text or ""))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def natural_key(value: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    return tuple(key)


if __name__ == "__main__":
    raise SystemExit(main())
