from __future__ import annotations

import json

import pytest

from northcua.agentkernel.context.domain.models import ContextSummaryDocumentV1, SummaryItem
from northcua.agentkernel.context.strategies import SummaryCompressionStrategy


def _strategy() -> SummaryCompressionStrategy:
    return SummaryCompressionStrategy(wrapper_characters=60, minimum_document_characters=220)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_compression_strategy_rejects_invalid_ratio_at_public_boundary(ratio: float) -> None:
    with pytest.raises(ValueError, match="minimum_reduction_ratio"):
        _strategy().minimum_source_characters(ratio)


def test_compression_strategy_rejects_negative_postprocess_reserve() -> None:
    with pytest.raises(ValueError, match="reserved_output_characters"):
        _strategy().resolve(
            retained_baseline_characters=1000,
            new_source_characters=500,
            minimum_reduction_ratio=0.25,
            reserved_output_characters=-1,
        )


def test_compression_strategy_preserves_baseline_and_compresses_only_new_source() -> None:
    contract = _strategy().resolve(
        retained_baseline_characters=1000,
        new_source_characters=400,
        minimum_reduction_ratio=0.25,
        reserved_output_characters=50,
    )

    assert contract.maximum_canonical_characters == 1300
    assert contract.global_final_ceiling_characters == 1300
    assert contract.maximum_model_canonical_characters == 1250
    assert contract.maximum_json_characters == 1190


def test_canonical_summary_omits_only_empty_optional_item_provenance() -> None:
    document = ContextSummaryDocumentV1(
        current_state=(SummaryItem(text="visible"),),
        verified_facts=(SummaryItem(text="supported", source_segment_ids=("segment-1",)),),
    )

    raw_json = document.canonical_prompt_text().split("\n", maxsplit=1)[1].rsplit("\n", maxsplit=1)[0]
    payload = json.loads(raw_json)

    assert set(payload) == {
        "schema_version",
        "goal",
        "constraints",
        "verified_facts",
        "decisions",
        "completed",
        "failures",
        "current_state",
        "open_items",
        "artifacts",
    }
    assert payload["current_state"] == [{"text": "visible"}]
    assert payload["verified_facts"] == [{"text": "supported", "source_segment_ids": ["segment-1"]}]
