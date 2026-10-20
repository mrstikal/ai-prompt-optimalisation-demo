"""Shared extraction of GEPA candidate records from native statistics."""

from __future__ import annotations

from collections.abc import Mapping


class CandidateStatsError(ValueError):
    """Raised when native GEPA statistics contain no inspectable candidate."""


def _records_from_stats(
    stats: Mapping[str, object],
) -> list[dict[str, object]]:
    candidates = stats.get("candidates")
    parents = stats.get("parents")
    scores = stats.get("val_aggregate_scores")
    discovery = stats.get("discovery_eval_counts")
    best_idx = stats.get("best_idx")
    if not isinstance(candidates, list):
        return []
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        instruction = candidate.get("predict")
        if not isinstance(instruction, str) or not instruction.strip():
            nested = candidate.get("instructions")
            if isinstance(nested, Mapping):
                instruction = nested.get("predict")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        parents_value = None
        if isinstance(parents, list) and index < len(parents):
            parent_items = parents[index]
            if isinstance(parent_items, list):
                parents_value = list(parent_items)
        records.append(
            {
                "candidate_index": index,
                "instructions": instruction,
                "parents": parents_value,
                "val_aggregate_score": (
                    scores[index]
                    if isinstance(scores, list) and index < len(scores)
                    else None
                ),
                "discovery_eval_count": (
                    discovery[index]
                    if isinstance(discovery, list)
                    and index < len(discovery)
                    else None
                ),
                "selected": index == best_idx,
            }
        )
    return records


def inspectable_candidate_records(
    stats: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract app-facing candidate records from a full stats payload."""
    if not isinstance(stats, Mapping):
        return [], "No candidate statistics are archived."
    detailed = stats.get("detailed_results")
    if not isinstance(detailed, Mapping):
        return [], "Candidate statistics do not contain detailed_results."
    records = _records_from_stats(detailed)
    if not records:
        return [], "Candidate statistics contain no inspectable instructions."
    return records, None


def gepa_candidate_records(
    stats: Mapping[str, object],
) -> list[dict[str, object]]:
    """Extract strict optimization records from native detailed results."""
    records = _records_from_stats(stats)
    if not records:
        raise CandidateStatsError(
            "GEPA stats contain no candidate instructions"
        )
    for record in records:
        record["candidate_id"] = f"candidate-{record['candidate_index']}"
        record["instructions"] = {"predict": record["instructions"]}
    return records


__all__ = [
    "CandidateStatsError",
    "gepa_candidate_records",
    "inspectable_candidate_records",
]
