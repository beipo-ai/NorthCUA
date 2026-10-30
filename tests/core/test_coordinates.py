from __future__ import annotations

from northcua.core.config import NorthCUAAppConfig
from northcua.core.coordinates import (
    CoordinateFrame,
    CoordinateMapper,
    CoordinatePoint,
    CoordinateTarget,
    apply_app_coordinate_environment,
    coordinate_policy_context_values,
    non_default_coordinate_policy_context_values,
    policy_from_config,
    resolve_coordinate_policy,
    restore_coordinate_environment,
)
from northcua.core.enums import CoordinateSpace


def test_coordinate_mapper_maps_image_pixels_to_absolute_runtime_point() -> None:
    point = CoordinateMapper.to_runtime_point(
        CoordinatePoint(x=25, y=40),
        policy=policy_from_config(
            NorthCUAAppConfig(
                agent_config_path="agent.yaml",
            ).desktop.coordinate_policy
        ),
        frame=CoordinateFrame(width=400, height=300, offset_x=100, offset_y=200),
    )

    assert point == CoordinatePoint(x=125, y=240)


def test_coordinate_mapper_maps_normalized_unit_to_runtime_pixels() -> None:
    point = CoordinateMapper.to_runtime_point(
        CoordinatePoint(x=0.5, y=0.25),
        policy=resolve_coordinate_policy(
            CoordinateTarget.DESKTOP,
            fallback_policy=policy_from_config(
                NorthCUAAppConfig(agent_config_path="agent.yaml").desktop.coordinate_policy
            ),
            context_values=coordinate_policy_context_values(
                CoordinateTarget.DESKTOP,
                policy_from_config(
                    NorthCUAAppConfig(
                        agent_config_path="agent.yaml",
                        desktop={"coordinate_policy": {"input_space": "normalized_unit"}},
                    ).desktop.coordinate_policy
                ),
            ),
        ),
        frame=CoordinateFrame(width=200, height=100),
    )

    assert point == CoordinatePoint(x=100, y=25)


def test_apply_app_coordinate_environment_exports_and_restores_values() -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        desktop={"coordinate_policy": {"input_space": "absolute_pixels", "clamp_to_bounds": False}},
    )
    environment = {
        "NORTHCUA_DESKTOP_COORDINATE_SPACE": "image_pixels",
        "NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS": "true",
    }

    snapshot = apply_app_coordinate_environment(app_config, environment=environment)

    assert environment["NORTHCUA_DESKTOP_COORDINATE_SPACE"] == "absolute_pixels"
    assert environment["NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS"] == "false"

    restore_coordinate_environment(snapshot, environment=environment)

    assert environment["NORTHCUA_DESKTOP_COORDINATE_SPACE"] == "image_pixels"
    assert environment["NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS"] == "true"


def test_resolve_coordinate_policy_prefers_context_over_environment() -> None:
    resolved = resolve_coordinate_policy(
        CoordinateTarget.DESKTOP,
        context_values=coordinate_policy_context_values(
            CoordinateTarget.DESKTOP,
            policy_from_config(
                NorthCUAAppConfig(
                    agent_config_path="agent.yaml",
                    desktop={"coordinate_policy": {"input_space": "normalized_1000", "clamp_to_bounds": False}},
                ).desktop.coordinate_policy
            ),
        ),
        environment={
            "NORTHCUA_DESKTOP_COORDINATE_SPACE": CoordinateSpace.ABSOLUTE_PIXELS.value,
            "NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS": "true",
        },
        fallback_policy=policy_from_config(NorthCUAAppConfig(agent_config_path="agent.yaml").desktop.coordinate_policy),
    )

    assert resolved.input_space == CoordinateSpace.NORMALIZED_1000
    assert resolved.clamp_to_bounds is False


def test_non_default_coordinate_policy_context_values_omits_default_desktop_pixels() -> None:
    values = non_default_coordinate_policy_context_values(
        CoordinateTarget.DESKTOP,
        policy_from_config(NorthCUAAppConfig(agent_config_path="agent.yaml").desktop.coordinate_policy),
    )

    assert values == {}


def test_non_default_coordinate_policy_context_values_keeps_normalized_desktop_policy() -> None:
    values = non_default_coordinate_policy_context_values(
        CoordinateTarget.DESKTOP,
        policy_from_config(
            NorthCUAAppConfig(
                agent_config_path="agent.yaml",
                desktop={"coordinate_policy": {"input_space": "normalized_1000", "clamp_to_bounds": True}},
            ).desktop.coordinate_policy
        ),
    )

    assert values == {
        "northcua.desktop.coordinate_space": "normalized_1000",
        "northcua.desktop.coordinate_clamp_to_bounds": "true",
    }
