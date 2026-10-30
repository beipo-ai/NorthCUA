"""CLI parser coverage for batch and sandbox commands."""

from __future__ import annotations

from northcua.cli.parallelism_args import apply_batch_parallelism_defaults
from northcua.cli.parser import build_parser
from northcua.cli.prepared_sandbox_args import prepared_policy_from_args


def test_batch_run_parser_accepts_manifest() -> None:
    args = build_parser().parse_args(["batch", "run", "--manifest", "tasks.yaml", "--dry-run", "--persist-token"])

    assert args.command_group == "batch"
    assert args.batch_command == "run"
    assert args.manifest == "tasks.yaml"
    assert args.dry_run is True
    assert args.persist_token is True


def test_batch_validate_parser_accepts_manifest() -> None:
    args = build_parser().parse_args(["batch", "validate", "--manifest", "tasks.yaml"])

    assert args.command_group == "batch"
    assert args.batch_command == "validate"
    assert args.manifest == "tasks.yaml"


def test_batch_resume_parser_accepts_runtime_overrides() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "resume",
            "--batch-id",
            "demo",
            "--controller-url",
            "ws://controller",
            "--token",
            "secret",
            "--controller-db",
            "controller.sqlite",
            "--concurrency",
            "4",
            "--create-concurrency",
            "1",
            "--submit-concurrency",
            "2",
            "--status-concurrency",
            "1",
            "--submit-jitter-seconds",
            "1.5",
            "--device-readiness-timeout-seconds",
            "600",
            "--infra-retry-backoff-seconds",
            "30,90,180",
        ]
    )

    assert args.command_group == "batch"
    assert args.batch_command == "resume"
    assert args.controller_url == "ws://controller"
    assert args.token == "secret"
    assert str(args.controller_db) == "controller.sqlite"
    assert args.concurrency == 4
    assert args.create_concurrency == 1
    assert args.submit_concurrency == 2
    assert args.status_concurrency == 1
    assert args.submit_jitter_seconds == 1.5
    assert args.device_readiness_timeout_seconds == 600
    assert args.infra_retry_backoff_seconds == "30,90,180"
    assert args.pre_resume_cleanup is True


def test_batch_parallelism_fills_unset_runtime_caps() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "run",
            "--manifest",
            "tasks.yaml",
            "--parallelism",
            "120",
            "--prepare-sandboxes",
        ]
    )

    apply_batch_parallelism_defaults(args)

    assert args.concurrency == 120
    assert args.create_concurrency == 24
    assert args.submit_concurrency == 36
    assert args.status_concurrency == 12
    assert args.prepare_buffer == 90
    assert args.prepare_concurrency == 36


def test_batch_parallelism_preserves_explicit_runtime_caps() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "resume",
            "--batch-id",
            "demo",
            "--parallelism",
            "120",
            "--concurrency",
            "80",
            "--create-concurrency",
            "10",
            "--prepare-sandboxes",
            "--prepare-buffer",
            "12",
        ]
    )

    apply_batch_parallelism_defaults(args)

    assert args.concurrency == 80
    assert args.create_concurrency == 10
    assert args.submit_concurrency == 36
    assert args.status_concurrency == 12
    assert args.prepare_buffer == 12
    assert args.prepare_concurrency == 36


def test_batch_resume_parser_can_disable_pre_resume_cleanup() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "resume",
            "--batch-id",
            "demo",
            "--no-pre-resume-cleanup",
            "--pre-resume-cleanup-provider",
            "fake",
        ]
    )

    assert args.batch_command == "resume"
    assert args.pre_resume_cleanup is False
    assert args.pre_resume_cleanup_provider == "fake"


def test_batch_run_parser_accepts_prepared_sandbox_flags() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "run",
            "--manifest",
            "tasks.yaml",
            "--prepare-sandboxes",
            "--prepare-depth",
            "fixture_setup",
            "--prepare-buffer",
            "8",
            "--prepare-concurrency",
            "2",
            "--prepare-initial-warmup-wait-seconds",
            "7",
            "--prepare-ttl-seconds",
            "120",
            "--prepare-claim-wait-seconds",
            "5",
            "--prepare-claim-pending-wait-seconds",
            "0.5",
            "--prepare-min-assignment-ttl-seconds",
            "900",
            "--prepare-assignment-ttl-margin-seconds",
            "60",
        ]
    )

    assert args.prepare_sandboxes is True
    assert args.prepare_depth == "fixture_setup"
    assert args.prepare_buffer == 8
    assert args.prepare_concurrency == 2
    assert args.prepare_initial_warmup_wait_seconds == 7
    assert args.prepare_ttl_seconds == 120
    assert args.prepare_claim_wait_seconds == 5
    assert args.prepare_claim_pending_wait_seconds == 0.5
    assert args.prepare_min_assignment_ttl_seconds == 900
    assert args.prepare_assignment_ttl_margin_seconds == 60


def test_prepared_sandbox_cli_default_claim_wait_is_one_hour() -> None:
    args = build_parser().parse_args(["batch", "run", "--manifest", "tasks.yaml", "--prepare-sandboxes"])

    policy = prepared_policy_from_args(args)

    assert policy.claim_wait_seconds == 3600.0
    assert policy.initial_warmup_wait_seconds == 5.0
    assert policy.claim_pending_wait_seconds == 0.0
    assert policy.min_assignment_ttl_seconds == 0.0
    assert policy.assignment_ttl_margin_seconds == 300.0


def test_batch_pause_and_drain_parser_accept_batch_id() -> None:
    pause = build_parser().parse_args(
        ["batch", "pause", "--batch-id", "demo", "--actor", "operator", "--reason", "investigate pressure"]
    )
    drain = build_parser().parse_args(
        ["batch", "drain", "--batch-id", "demo", "--actor", "operator", "--reason", "finish active only"]
    )
    cancel = build_parser().parse_args(["batch", "cancel", "--batch-id", "demo", "--no-graceful-cleanup"])

    assert pause.batch_command == "pause"
    assert pause.actor == "operator"
    assert pause.reason == "investigate pressure"
    assert drain.batch_command == "drain"
    assert drain.actor == "operator"
    assert drain.reason == "finish active only"
    assert cancel.batch_command == "cancel"
    assert cancel.graceful_cleanup is False


def test_batch_maintenance_parser_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "maintenance",
            "repair-stale",
            "--batch-id",
            "demo",
            "--older-than-seconds",
            "120",
        ]
    )

    assert args.command_group == "batch"
    assert args.batch_command == "maintenance"
    assert args.maintenance_command == "repair-stale"
    assert args.apply is False
    assert args.older_than_seconds == 120


def test_batch_maintenance_terminate_parser_scopes_case() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "maintenance",
            "terminate-sandboxes",
            "--batch-id",
            "demo",
            "--case-id",
            "case-1",
            "--provider",
            "e2b",
            "--force-active",
            "--terminate-concurrency",
            "12",
        ]
    )

    assert args.batch_command == "maintenance"
    assert args.maintenance_command == "terminate-sandboxes"
    assert args.case_id == "case-1"
    assert args.force_active is True
    assert args.terminate_concurrency == 12
    assert args.apply is False


def test_batch_maintenance_sweep_provider_orphans_parser_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "maintenance",
            "sweep-provider-orphans",
            "--batch-id",
            "demo",
            "--provider",
            "e2b",
            "--include-connect-existing",
            "--terminate-concurrency",
            "6",
        ]
    )

    assert args.batch_command == "maintenance"
    assert args.maintenance_command == "sweep-provider-orphans"
    assert args.provider == "e2b"
    assert args.include_connect_existing is True
    assert args.terminate_concurrency == 6
    assert args.apply is False


def test_batch_maintenance_cleanup_controller_residue_parser_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "batch",
            "maintenance",
            "cleanup-controller-residue",
            "--batch-id",
            "demo",
            "--batch-db",
            "batch.sqlite",
            "--controller-db",
            "controller.sqlite",
            "--no-protect-prepared",
            "--include-current-active",
            "--include-prepared",
            "--skip-provider",
        ]
    )

    assert args.batch_command == "maintenance"
    assert args.maintenance_command == "cleanup-controller-residue"
    assert args.apply is False
    assert args.protect_prepared is False
    assert args.include_current_active is True
    assert args.include_prepared is True
    assert args.skip_provider is True


def test_sandbox_probe_parser_accepts_provider() -> None:
    args = build_parser().parse_args(["sandbox", "probe", "--provider", "fake", "--template", "ubuntu"])

    assert args.command_group == "sandbox"
    assert args.sandbox_command == "probe"
    assert args.provider == "fake"


def test_sandbox_list_parser_accepts_json() -> None:
    args = build_parser().parse_args(["sandbox", "list", "--json"])

    assert args.command_group == "sandbox"
    assert args.sandbox_command == "list"
    assert args.json is True
