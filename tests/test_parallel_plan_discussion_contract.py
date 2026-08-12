from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents" / "skills" / "parallel-plan-discussion" / "SKILL.md"
TEMPLATE = (
    ROOT
    / ".agents"
    / "skills"
    / "parallel-plan-discussion"
    / "references"
    / "task-message-template.md"
)


def skill_sources() -> tuple[str, str, str]:
    skill = SKILL.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    return skill, template, f"{skill}\n{template}"


def test_parallel_plan_discussion_removes_stale_orchestration_paths() -> None:
    _, _, source = skill_sources()

    for forbidden in (
        "delegate-codex",
        "agmsg",
        "gwq",
        ".claude/skills",
        "wait-pane.sh",
        "manual polling",
        "unbounded polling",
        "poll",
        "nudge",
        "sleep",
    ):
        assert forbidden not in source


def test_parallel_plan_discussion_has_herdr_only_council_contract() -> None:
    _, template, source = skill_sources()

    for required in (
        "HERDR_ENV=1",
        "herdr --skill",
        "Herdr-managed worktree",
        "agent names",
        "pane IDs",
        "herdr agent prompt",
        "directly to each peer",
        "herdr agent get",
        "herdr agent read",
        "herdr agent wait",
        "bounded",
        "exactly two rounds",
        "Round 1",
        "Round 2",
        "chair",
        "unified spec",
        "herdr notification show",
        "explicit user authorization",
    ):
        assert required in source

    assert "herdr agent prompt" in template
    assert "Round 1" in template
    assert "Round 2" in template
    assert "unified spec" in template
    assert "only the shared drafts directory" in template

    assert "herdr --help" not in source
    assert "herdr pane split" not in source
    assert "herdr tab create" not in source
    assert "herdr workspace create" not in source


def test_coordinator_identity_is_recorded_and_reports_use_it() -> None:
    _, template, source = skill_sources()

    for required in (
        "coordinator agent name",
        "coordinator pane ID",
        "recorded coordinator agent name or pane ID",
    ):
        assert required in source

    for required in (
        "<coordinator-agent-name>",
        "<coordinator-pane-id>",
        "<coordinator-agent-name-or-pane-id>",
    ):
        assert required in template

    assert "`main`" not in source
    assert "`main`" not in template


def test_only_agent_wait_uses_a_finite_timeout() -> None:
    _, _, source = skill_sources()

    assert "immediate `herdr agent get` and `herdr agent read`" in source
    assert "only `herdr agent wait` uses a finite timeout" in source
    assert (
        "herdr agent get`, `herdr agent read`, and `herdr agent wait`, each with a finite timeout"
        not in source
    )


def test_notification_is_a_coordinator_user_ui_fallback() -> None:
    _, template, source = skill_sources()

    assert "UI fallback for the coordinator/user" in source
    assert "does not send a participant message" in source
    assert "UI fallback for the coordinator/user" in template
    assert "participant receives the coordinator's fallback" not in template
