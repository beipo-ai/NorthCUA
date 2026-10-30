"""Keep execution/evaluation errors out of both score sum and denominator."""

import math
from collections.abc import Iterable, Mapping


def score_record(*, task_id: str, raw_score=None, error: str = "", terminal_reason: str = "") -> dict:
    invalid = bool(error) or terminal_reason in {"empty_response", "content_filter", "runtime_error"}
    score = None
    if not invalid:
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            error = "Evaluator did not return a numeric score"
            invalid = True
        elif not math.isfinite(raw_score) or not 0 <= raw_score <= 1:
            error = "Evaluator score is outside [0, 1] or nonfinite"
            invalid = True
        else:
            score = float(raw_score)
    return {
        "task_id": task_id,
        "raw_score": raw_score,
        "score": score,
        "score_valid": not invalid,
        "error": error,
        "terminal_reason": terminal_reason,
    }


def summarize_scores(records: Iterable[Mapping]) -> dict:
    rows = list(records)
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Select one current attempt per logical task before aggregation")
    values = []
    for row in rows:
        if row.get("score_valid") is True:
            checked = score_record(
                task_id=row["task_id"],
                raw_score=row.get("score"),
                error=row.get("error", ""),
                terminal_reason=row.get("terminal_reason", ""),
            )
            if not checked["score_valid"]:
                raise ValueError("Invalid row marked score_valid")
            values.append(checked["score"])
    total = math.fsum(values)
    full = sum(value == 1 for value in values)
    return {
        "total_tasks": len(rows),
        "valid_tasks": len(values),
        "excluded_tasks": len(rows) - len(values),
        "total_score": total,
        "mean_score": total / len(values) if values else None,
        "full_score_tasks": full,
        "binary_rate": full / len(values) if values else None,
        "valid_zero_tasks": values.count(0),
    }
