"""Inspect a real web-agent run artifact directory for trace/SoM issues."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Finding:
    severity: str
    title: str
    detail: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows.append(payload if isinstance(payload, dict) else {"payload": payload})
    return rows


def _count_images(value: object) -> tuple[int, int]:
    count = 0
    data_urls = 0
    if isinstance(value, dict):
        typ = value.get("type")
        if (isinstance(typ, str) and typ in {"input_image", "image_url"}) or "image_url" in value:
            count += 1
            url = value.get("image_url") or value.get("url") or ""
            if isinstance(url, dict):
                url = url.get("url", "")
            if isinstance(url, str) and url.lower().startswith("data:image"):
                data_urls += 1
        for child in value.values():
            nested_count, nested_data_urls = _count_images(child)
            count += nested_count
            data_urls += nested_data_urls
    elif isinstance(value, list):
        for child in value:
            nested_count, nested_data_urls = _count_images(child)
            count += nested_count
            data_urls += nested_data_urls
    return count, data_urls


def _collect_image_urls(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        typ = value.get("type")
        if (isinstance(typ, str) and typ in {"input_image", "image_url"}) or "image_url" in value:
            url = value.get("image_url") or value.get("url") or ""
            if isinstance(url, dict):
                url = url.get("url", "")
            if isinstance(url, str) and url:
                urls.append(url)
        for child in value.values():
            urls.extend(_collect_image_urls(child))
    elif isinstance(value, list):
        for child in value:
            urls.extend(_collect_image_urls(child))
    return urls


def _contains_model_visible_path(value: object) -> bool:
    encoded = json.dumps(value, ensure_ascii=False).lower()
    return any(token in encoded for token in ("/screenshots/", "\\screenshots\\", ".png"))


def _contains_path_or_data_url(value: object) -> bool:
    encoded = json.dumps(value, ensure_ascii=False).lower()
    return any(token in encoded for token in ("data:image", "/screenshots/", "\\screenshots\\", ".png"))


def _extract_ref_id(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        ref = value.get("ref")
        return ref if isinstance(ref, str) else ""
    return ""


def _declared_tool_names(tools: object) -> set[str]:
    """Return tool names from Responses- and Chat-style tool descriptors."""

    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        function = tool.get("function")
        if not isinstance(name, str) and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _called_tool_names(messages: object) -> list[str]:
    """Return assistant function names embedded in provider request history."""

    names: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "function_call":
                name = value.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
            tool_calls = value.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    name = function.get("name") if isinstance(function, dict) else call.get("name")
                    if isinstance(name, str) and name:
                        names.append(name)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(messages)
    return names


def _observation_som_ref_alignment_issues(
    *,
    row_index: int,
    tool_name: str,
    observation: dict[str, object],
    observation_error: object,
) -> list[str]:
    snapshot = observation.get("snapshot")
    screenshot = observation.get("screenshot")
    if not isinstance(snapshot, dict) or not isinstance(screenshot, dict):
        return []
    annotation = screenshot.get("annotation")
    if not isinstance(annotation, dict) or annotation.get("kind") != "set_of_mark":
        return []

    refs = snapshot.get("refs")
    marks = annotation.get("marks")
    ref_ids = [_extract_ref_id(ref) for ref in refs] if isinstance(refs, list) else []
    mark_refs = [_extract_ref_id(mark) for mark in marks] if isinstance(marks, list) else []
    mark_labels = [str(mark.get("label") or "") for mark in marks] if isinstance(marks, list) else []

    issues: list[str] = []
    if ref_ids != mark_refs:
        issues.append(
            f"row={row_index} tool={tool_name} refs={len(ref_ids)} marks={len(mark_refs)} "
            f"first_refs={ref_ids[:5]} first_marks={mark_refs[:5]}"
        )
    elif mark_labels and mark_labels != ref_ids:
        issues.append(
            f"row={row_index} tool={tool_name} labels do not equal refs "
            f"first_refs={ref_ids[:5]} first_labels={mark_labels[:5]}"
        )

    page = observation.get("page")
    page_url = ""
    if isinstance(page, dict) and isinstance(page.get("url"), str):
        page_url = str(page.get("url"))
    if (
        not ref_ids
        and not mark_refs
        and screenshot.get("image_attached") is True
        and page_url
        and page_url != "about:blank"
        and not observation_error
    ):
        issues.append(f"row={row_index} tool={tool_name} has image but zero refs/marks without observation_error")
    return issues


def _qa_llm_trace(root: Path, findings: list[Finding]) -> dict[str, object]:
    rows = _load_jsonl(root / "llm_trace.jsonl")
    image_counts: list[int] = []
    data_image_counts: list[int] = []
    wrapper_ok = True
    wrapper_errors: list[str] = []
    session_steps: list[int] = []
    browser_tools: list[str] = []
    non_data_image_refs: list[str] = []
    undeclared_tool_calls: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_errors: list[str] = []
        if row.get("source") != "web":
            row_errors.append("source")
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            row_errors.append("session_id")
        session_step = row.get("session_step")
        if isinstance(session_step, int) and session_step > 0:
            session_steps.append(session_step)
        else:
            row_errors.append("session_step")
        round_num = row.get("round_num")
        if not isinstance(round_num, int) or round_num <= 0:
            row_errors.append("round_num")
        round_step = row.get("round_step")
        if not isinstance(round_step, int) or round_step <= 0:
            row_errors.append("round_step")
        if not isinstance(row.get("request"), str):
            row_errors.append("request")
        if not isinstance(row.get("runtime_cursor"), dict):
            row_errors.append("runtime_cursor")
        trace = row.get("llm_trace")
        if not isinstance(trace, dict):
            findings.append(Finding("P0", "llm_trace row missing nested llm_trace", f"row={index}"))
            continue
        request = trace.get("request")
        if not isinstance(request, dict):
            row_errors.append("llm_trace.request")
            request = {}
        image_count, data_image_count = _count_images(request)
        image_counts.append(image_count)
        data_image_counts.append(data_image_count)
        for image_url in _collect_image_urls(request):
            if not image_url.lower().startswith("data:image"):
                non_data_image_refs.append(f"row={index}:{image_url[:80]}")
        tools = request.get("tools", []) if isinstance(request, dict) else []
        if isinstance(tools, list):
            browser_tools.extend(
                str(tool.get("name"))
                for tool in tools
                if isinstance(tool, dict) and str(tool.get("name", "")).startswith("browser_")
            )
        declared_tools = _declared_tool_names(tools)
        for request_field in ("messages", "input"):
            for called_tool in _called_tool_names(request.get(request_field)):
                if called_tool not in declared_tools:
                    undeclared_tool_calls.append(f"row={index}:{request_field}:{called_tool}")
        if _contains_model_visible_path(request):
            findings.append(Finding("P0", "model request contains local artifact path", f"row={index}"))
        if row_errors:
            wrapper_errors.append(f"row={index}:{','.join(row_errors)}")
    if any(current <= previous for previous, current in zip(session_steps, session_steps[1:], strict=False)):
        wrapper_ok = False
        wrapper_errors.append("session_step_order")
    if wrapper_errors:
        wrapper_ok = False
    if rows and not wrapper_ok:
        findings.append(
            Finding(
                "P0",
                "llm_trace wrapper fields are inconsistent",
                "Expected Desktop-style Web wrapper fields on every row: " + "; ".join(wrapper_errors[:10]),
            )
        )
    if non_data_image_refs:
        findings.append(
            Finding(
                "P1",
                "llm_trace image blocks are not raw data URLs",
                "; ".join(non_data_image_refs[:10]),
            )
        )
    if browser_tools:
        findings.append(
            Finding("P0", "browser_* tools leaked into model tool surface", ", ".join(sorted(set(browser_tools))))
        )
    if undeclared_tool_calls:
        findings.append(
            Finding(
                "P0",
                "prompt history contains undeclared function calls",
                ", ".join(undeclared_tool_calls[:20]),
            )
        )
    return {
        "row_count": len(rows),
        "image_counts": image_counts,
        "data_image_counts": data_image_counts,
        "non_data_image_refs": non_data_image_refs,
        "browser_tool_count": len(browser_tools),
        "undeclared_tool_calls": undeclared_tool_calls,
    }


def _qa_web_trace(root: Path, findings: list[Finding]) -> dict[str, object]:
    rows = _load_jsonl(root / "web_agent_trace.jsonl")
    tool_summaries: list[str] = []
    tool_outputs_with_paths: list[str] = []
    removed_top_level_fields: list[str] = []
    som_ref_alignment_issues: list[str] = []
    post_observations = 0
    tool_result_images = 0
    removed_field_names = ("action", "tool_input", "tool_output")
    for index, row in enumerate(rows, start=1):
        removed_fields = [field for field in removed_field_names if field in row]
        if removed_fields:
            removed_top_level_fields.append(f"row {index}: {', '.join(removed_fields)}")
        tool_summaries.append(str(row.get("tool_summary") or ""))
        for tool in row.get("tools", []):
            if not isinstance(tool, dict):
                continue
            output = tool.get("tool_output")
            tool_name = str(tool.get("tool_name") or "")
            if isinstance(output, dict):
                if _contains_path_or_data_url(output):
                    tool_outputs_with_paths.append(tool_name)
                if isinstance(output.get("observation"), dict):
                    post_observations += 1
                    som_ref_alignment_issues.extend(
                        _observation_som_ref_alignment_issues(
                            row_index=index,
                            tool_name=tool_name,
                            observation=output["observation"],
                            observation_error=output.get("observation_error"),
                        )
                    )
                if output.get("screenshot_url"):
                    tool_result_images += 1
    if tool_outputs_with_paths:
        findings.append(
            Finding(
                "P0",
                "model-visible tool output contains path or raw data URL",
                ", ".join(tool_outputs_with_paths),
            )
        )
    if removed_top_level_fields:
        findings.append(
            Finding(
                "P1",
                "removed top-level web trace fields present",
                "; ".join(removed_top_level_fields[:10]),
            )
        )
    if som_ref_alignment_issues:
        findings.append(
            Finding(
                "P1",
                "model-visible SoM refs do not match annotation marks",
                "; ".join(som_ref_alignment_issues[:10]),
            )
        )
    return {
        "row_count": len(rows),
        "tool_summaries": tool_summaries,
        "post_observation_count": post_observations,
        "tool_result_image_count": tool_result_images,
        "tool_outputs_with_path_or_data_url": tool_outputs_with_paths,
        "removed_top_level_fields": removed_top_level_fields,
        "som_ref_alignment_issues": som_ref_alignment_issues,
    }


def _qa_som(root: Path, findings: list[Finding]) -> dict[str, object]:
    annotation_files = sorted((root / "screenshots").glob("*.annotation.json"))
    summaries: list[dict[str, object]] = []
    for path in annotation_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        marks = payload.get("marks", [])
        screenshot_size = _annotation_screenshot_size(path, payload)
        invalid = []
        for mark in marks if isinstance(marks, list) else []:
            if not isinstance(mark, dict):
                invalid.append(mark)
                continue
            css = mark.get("bbox_css", {})
            image = mark.get("bbox_image", {})
            if (
                not mark.get("ref")
                or not isinstance(css, dict)
                or not isinstance(image, dict)
                or float(css.get("width") or 0) <= 0
                or float(css.get("height") or 0) <= 0
                or float(image.get("width") or 0) <= 0
                or float(image.get("height") or 0) <= 0
            ):
                invalid.append(mark)
                continue
            if screenshot_size is not None and _bbox_exceeds_image_bounds(image, screenshot_size):
                invalid.append(mark)
        if invalid:
            findings.append(Finding("P1", "SoM annotation contains invalid mark geometry", path.name))
        summaries.append(
            {
                "file": path.name,
                "mark_count": len(marks) if isinstance(marks, list) else 0,
                "invalid_mark_count": len(invalid),
                "truncated": bool(payload.get("truncated")),
                "omitted_count": int(payload.get("omitted_count") or 0),
                "has_source_path": bool(payload.get("source_screenshot_path")),
                "has_overlay_path": bool(payload.get("overlay_screenshot_path")),
                "screenshot_size": list(screenshot_size) if screenshot_size is not None else None,
            }
        )
    if not summaries:
        findings.append(Finding("P1", "no SoM annotation files found", str(root / "screenshots")))
    elif not any(summary["mark_count"] for summary in summaries):
        findings.append(Finding("P1", "SoM annotation files contain no marks", str(root / "screenshots")))
    return {"annotation_count": len(summaries), "annotations": summaries}


def _annotation_screenshot_size(path: Path, payload: dict[str, object]) -> tuple[int, int] | None:
    """Return the overlay/source PNG dimensions for one annotation sidecar."""

    candidates: list[Path] = []
    if path.name.endswith(".annotation.json"):
        candidates.append(path.with_name(path.name[: -len(".annotation.json")] + ".png"))
    for key in ("overlay_screenshot_path", "source_screenshot_path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                return image.size
        except Exception:
            continue
    return None


def _bbox_exceeds_image_bounds(bbox: dict[str, object], image_size: tuple[int, int]) -> bool:
    """Validate image-space bounding boxes against the screenshot dimensions."""

    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    width = float(bbox.get("width") or 0)
    height = float(bbox.get("height") or 0)
    image_width, image_height = image_size
    epsilon = 0.5
    return x < -epsilon or y < -epsilon or x + width > image_width + epsilon or y + height > image_height + epsilon


def _qa_artifact_links(root: Path, result: dict[str, Any], findings: list[Finding]) -> dict[str, object]:
    rows = _load_jsonl(root / "web_agent_trace.jsonl")
    result_screenshots = [str(item) for item in result.get("screenshots", []) if item]
    missing_result_screenshots = [path for path in result_screenshots if not Path(path).exists()]
    if missing_result_screenshots:
        findings.append(
            Finding(
                "P1",
                "result.screenshots references missing files",
                ", ".join(missing_result_screenshots[:5]),
            )
        )

    post_action_rows: list[dict[str, object]] = []
    missing_step_links: list[str] = []
    missing_result_links: list[str] = []
    current_points_to_tool_image: list[str] = []
    legacy_tool_image_rows: list[str] = []
    invalid_tool_artifacts: list[str] = []
    tool_artifact_details: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        tools = row.get("tools", [])
        has_post_image = False
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                output = tool.get("tool_output")
                result = output.get("result") if isinstance(output, dict) else None
                if isinstance(output, dict) and (
                    bool(output.get("screenshot_url"))
                    or (isinstance(result, dict) and bool(result.get("image_attached")))
                ):
                    has_post_image = True
                    break
        if not has_post_image:
            continue
        raw_tool_artifacts = row.get("tool_artifacts", [])
        has_typed_tool_artifacts = isinstance(raw_tool_artifacts, list) and bool(raw_tool_artifacts)
        if isinstance(raw_tool_artifacts, list):
            for artifact_index, artifact in enumerate(raw_tool_artifacts, start=1):
                issue = _validate_tool_artifact(row_index=index, artifact_index=artifact_index, artifact=artifact)
                if issue:
                    invalid_tool_artifacts.append(issue)
                tool_artifact_details.append(
                    _tool_artifact_detail(row_index=index, artifact_index=artifact_index, artifact=artifact)
                )
        tool_artifact_paths = [
            str(artifact.get("selected_image_path") or "")
            for artifact in raw_tool_artifacts
            if isinstance(artifact, dict) and artifact.get("selected_image_path")
        ]
        if not tool_artifact_paths:
            legacy_screenshot_path = str(row.get("screenshot_path") or "")
            if Path(legacy_screenshot_path).name.startswith("tool_"):
                tool_artifact_paths = [legacy_screenshot_path]
                legacy_tool_image_rows.append(f"row={index}:{Path(legacy_screenshot_path).name}")
        post_action_rows.append(
            {
                "row": index,
                "current_screenshot_path": str(row.get("screenshot_path") or ""),
                "tool_artifact_paths": tool_artifact_paths,
            }
        )
        if not tool_artifact_paths or not any(Path(path).exists() for path in tool_artifact_paths):
            missing_step_links.append(f"row={index}:{tool_artifact_paths}")
            continue
        for tool_path in tool_artifact_paths:
            if tool_path not in result_screenshots:
                missing_result_links.append(f"row={index}:{tool_path}")
        current_name = Path(str(row.get("screenshot_path") or "")).name
        if has_typed_tool_artifacts and current_name.startswith("tool_"):
            current_points_to_tool_image.append(f"row={index}:{current_name}")
    if missing_step_links:
        findings.append(
            Finding("P1", "post-action image row is missing a persisted screenshot_path", ", ".join(missing_step_links))
        )
    if missing_result_links:
        findings.append(
            Finding("P1", "post-action image row is missing from result.screenshots", ", ".join(missing_result_links))
        )
    if current_points_to_tool_image:
        findings.append(
            Finding(
                "P1",
                "step current screenshot points at tool-result image",
                ", ".join(current_points_to_tool_image),
            )
        )
    if invalid_tool_artifacts:
        findings.append(
            Finding(
                "P1",
                "post-action tool artifact attribution is incomplete",
                "; ".join(invalid_tool_artifacts[:10]),
            )
        )
    return {
        "result_screenshot_count": len(result_screenshots),
        "missing_result_screenshots": missing_result_screenshots,
        "post_action_rows": post_action_rows,
        "tool_artifact_details": tool_artifact_details,
        "legacy_tool_image_rows": legacy_tool_image_rows,
        "invalid_tool_artifacts": invalid_tool_artifacts,
    }


def _tool_artifact_detail(*, row_index: int, artifact_index: int, artifact: object) -> dict[str, object]:
    if not isinstance(artifact, dict):
        return {"row": row_index, "artifact": artifact_index, "valid": False, "reason": "not_object"}
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    selected_path = str(artifact.get("selected_image_path") or "")
    selected_file = Path(selected_path) if selected_path else None
    return {
        "row": row_index,
        "artifact": artifact_index,
        "tool_name": str(artifact.get("tool_name") or ""),
        "tool_call_id": str(artifact.get("tool_call_id") or ""),
        "selected_image_kind": str(artifact.get("selected_image_kind") or ""),
        "selected_image_path": selected_path,
        "raw_image_path": str(artifact.get("raw_image_path") or ""),
        "som_image_path": str(artifact.get("som_image_path") or ""),
        "selected_image_exists": bool(selected_file and selected_file.is_file()),
        "selected_image_sha256": _sha256_file(selected_file) if selected_file and selected_file.is_file() else "",
        "attribution_source": str(metadata.get("attribution_source") or ""),
        "invocation_id": metadata.get("invocation_id"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tool_artifact(*, row_index: int, artifact_index: int, artifact: object) -> str:
    prefix = f"row={row_index}:artifact={artifact_index}"
    if not isinstance(artifact, dict):
        return f"{prefix}:not_object"
    tool_name = artifact.get("tool_name")
    tool_call_id = artifact.get("tool_call_id")
    selected_kind = str(artifact.get("selected_image_kind") or "")
    raw_path = str(artifact.get("raw_image_path") or "")
    som_path = str(artifact.get("som_image_path") or "")
    selected_path = str(artifact.get("selected_image_path") or "")
    metadata = artifact.get("metadata")
    missing: list[str] = []
    if not isinstance(tool_name, str) or not tool_name:
        missing.append("tool_name")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        missing.append("tool_call_id")
    if selected_kind not in {"raw", "som", "none"}:
        missing.append("selected_image_kind")
    if not isinstance(metadata, dict):
        missing.append("metadata")
        metadata = {}
    if metadata.get("source") != "web_tool_result":
        missing.append("metadata.source")
    attribution_source = str(metadata.get("attribution_source") or "")
    if attribution_source not in {"explicit_path", "image_hash", "none"}:
        missing.append("metadata.attribution_source")
    if missing:
        return f"{prefix}:missing_or_invalid={','.join(missing)}"
    if selected_kind == "none":
        if selected_path or raw_path or som_path:
            return f"{prefix}:none_has_image_path"
        return ""
    if attribution_source == "none":
        return f"{prefix}:image_without_attribution_source"
    if selected_kind == "raw":
        if not raw_path or selected_path != raw_path:
            return f"{prefix}:raw_selected_path_mismatch"
        if not Path(raw_path).is_file():
            return f"{prefix}:raw_path_missing"
        return ""
    if selected_kind == "som":
        if not som_path or selected_path != som_path:
            return f"{prefix}:som_selected_path_mismatch"
        if not Path(som_path).is_file():
            return f"{prefix}:som_path_missing"
        if raw_path and not Path(raw_path).is_file():
            return f"{prefix}:raw_path_missing"
        return ""
    return ""


def run_qa(root: Path) -> dict[str, object]:
    findings: list[Finding] = []
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    if result.get("status") != "success":
        findings.append(Finding("P0", "web agent result is not success", str(result.get("status"))))
    report = {
        "artifact_root": str(root),
        "result_status": result.get("status"),
        "final_answer": result.get("final_answer", ""),
        "llm_trace": _qa_llm_trace(root, findings),
        "web_trace": _qa_web_trace(root, findings),
        "som": _qa_som(root, findings),
        "artifact_links": _qa_artifact_links(root, result, findings),
        "findings": [asdict(finding) for finding in findings],
        "status": "pass" if not any(f.severity in {"P0", "P1"} for f in findings) else "fail",
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root")
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-finding", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_qa(Path(args.artifact_root))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if args.fail_on_finding and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
