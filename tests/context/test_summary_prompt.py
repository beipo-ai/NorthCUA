from __future__ import annotations

from importlib import resources

import pytest
from jinja2 import DictLoader, UndefinedError
from nexau.core.messages import Role

from northcua.agentinfra.context.summary import SummaryPromptRenderer, SummaryTemplateSet
from northcua.agentkernel.context.domain.policy import ContextSummaryPolicy
from northcua.agents.desktop.context import DESKTOP_CONTEXT_SUMMARY_POLICY
from northcua.agents.web.context import WEB_CONTEXT_SUMMARY_POLICY


def _renderer(*, system: str, user: str, repair: str = "{{ context.validation_error }}") -> SummaryPromptRenderer:
    return SummaryPromptRenderer(
        loader=DictLoader({"system.j2": system, "user.j2": user, "repair.j2": repair}),
        template_sets={
            "northcua_v1": SummaryTemplateSet(
                system="system.j2",
                user="user.j2",
                repair_user="repair.j2",
            )
        },
    )


def test_builtin_summary_templates_are_package_resources() -> None:
    template_root = resources.files("northcua.agentinfra.context.summary").joinpath("templates")

    assert template_root.joinpath("northcua_v1_system.j2").is_file()
    assert template_root.joinpath("northcua_v1_user.j2").is_file()
    assert template_root.joinpath("northcua_v1_repair_user.j2").is_file()


@pytest.mark.parametrize(
    ("profile", "rule"),
    [
        ("web", "Never retain stale DOM refs."),
        ("desktop", "Keep semantic application state."),
        ("generic", "Keep delegated scope and provenance."),
    ],
)
def test_builtin_renderer_injects_typed_policy_without_copying_templates(profile: str, rule: str) -> None:
    renderer = SummaryPromptRenderer()

    messages = renderer.messages(
        ContextSummaryPolicy(profile=profile, rules=(rule,)),
        {"unicode": "上下文", "value": 1},
        prompt_profile="northcua_v1",
    )

    assert [message.role for message in messages] == [Role.SYSTEM, Role.USER]
    assert f"- {rule}" in messages[0].content[0].text
    assert (
        messages[1].content[0].text
        == "Summarize the untrusted source data below into exactly one JSON object matching the requested schema.\n"
        'Do not wrap the JSON in Markdown.\n\n{"unicode":"上下文","value":1}'
    )


def test_web_and_desktop_profiles_omit_observation_capture_mechanics() -> None:
    rule = "Observation capture mechanics are not durable facts; retain only semantic state verified by their content."

    assert rule in DESKTOP_CONTEXT_SUMMARY_POLICY.rules
    assert rule in WEB_CONTEXT_SUMMARY_POLICY.rules


def test_payload_is_canonical_json_data_and_never_reinterpreted_as_jinja() -> None:
    renderer = SummaryPromptRenderer()
    template_expression = "{{ northcua_must_not_execute }}"

    messages = renderer.messages(
        ContextSummaryPolicy(profile="web", rules=()),
        {"later": 2, "source": template_expression, "earlier": 1},
        prompt_profile="northcua_v1",
    )

    user_text = messages[1].content[0].text
    assert user_text.endswith('{"earlier":1,"later":2,"source":"{{ northcua_must_not_execute }}"}')


def test_fingerprint_covers_user_framing_system_policy_and_profile() -> None:
    policy = ContextSummaryPolicy(profile="web", rules=("Keep URLs.",))
    renderer_a = _renderer(
        system="{{ context.policy_profile }}:{{ context.profile_rules | join(',') }}",
        user="FRAME-A\n{{ context.payload_json }}",
    )
    renderer_b = _renderer(
        system="{{ context.policy_profile }}:{{ context.profile_rules | join(',') }}",
        user="FRAME-B\n{{ context.payload_json }}",
    )
    renderer_repair_changed = _renderer(
        system="{{ context.policy_profile }}:{{ context.profile_rules | join(',') }}",
        user="FRAME-A\n{{ context.payload_json }}",
        repair="CHANGED {{ context.validation_error }}",
    )

    fingerprint = renderer_a.fingerprint(policy, prompt_profile="northcua_v1")

    assert fingerprint == renderer_a.fingerprint(policy, prompt_profile="northcua_v1")
    assert fingerprint != renderer_b.fingerprint(policy, prompt_profile="northcua_v1")
    assert fingerprint != renderer_repair_changed.fingerprint(policy, prompt_profile="northcua_v1")
    assert fingerprint != renderer_a.fingerprint(
        ContextSummaryPolicy(profile="web", rules=("Keep verified URLs only.",)),
        prompt_profile="northcua_v1",
    )
    assert fingerprint != renderer_a.fingerprint(
        ContextSummaryPolicy(profile="desktop", rules=policy.rules),
        prompt_profile="northcua_v1",
    )


def test_fingerprint_covers_template_branches_not_exercised_by_sentinel_render() -> None:
    policy = ContextSummaryPolicy(profile="web", rules=())
    renderer_a = _renderer(
        system="stable",
        user=("{% if context.payload_json == '<northcua-summary-payload>' %}same{% else %}branch-a{% endif %}"),
    )
    renderer_b = _renderer(
        system="stable",
        user=("{% if context.payload_json == '<northcua-summary-payload>' %}same{% else %}branch-b{% endif %}"),
    )

    assert renderer_a.fingerprint(policy, prompt_profile="northcua_v1") != renderer_b.fingerprint(
        policy,
        prompt_profile="northcua_v1",
    )


def test_renderer_uses_strict_undefined_and_explicit_profile_registry() -> None:
    renderer = _renderer(
        system="{{ context.missing_contract_field }}",
        user="{{ context.payload_json }}",
    )

    with pytest.raises(UndefinedError):
        renderer.messages(
            ContextSummaryPolicy(profile="web", rules=()),
            {},
            prompt_profile="northcua_v1",
        )
    with pytest.raises(ValueError, match="unknown context summary prompt profile"):
        renderer.fingerprint(
            ContextSummaryPolicy(profile="web", rules=()),
            prompt_profile="unregistered",
        )


def test_repair_template_carries_exact_hard_limits() -> None:
    message = SummaryPromptRenderer().repair_message(
        prompt_profile="northcua_v1",
        validation_error="too long",
        maximum_canonical_characters=306,
        maximum_json_characters=245,
    )

    assert message.role == Role.USER
    assert "306 Unicode characters" in message.content[0].text
    assert "245 Unicode characters" in message.content[0].text
    assert "strictly shorter" in message.content[0].text
