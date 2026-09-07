from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "AGENTS.md"
MODEL_SKILL = ROOT / ".agents/skills/design-generators-model-conversion/SKILL.md"
DOCUMENTATION_SKILL = ROOT / ".agents/skills/design-generators-documentation/SKILL.md"


def test_migrated_guidance_keeps_critical_owners() -> None:
    core = CORE.read_text(encoding="utf-8")
    model_skill = MODEL_SKILL.read_text(encoding="utf-8")
    documentation_skill = DOCUMENTATION_SKILL.read_text(encoding="utf-8")

    semantic_rule = (
        "Within function bodies, separate semantic units (configuration branches, "
        "submodule construction, transformations, and return preparation) with "
        "single blank lines; always leave a blank line after a raise block when "
        "ordinary code follows; leave one blank line after an if, for, while, try, "
        "or with suite when ordinary code follows at the enclosing indentation."
    )
    reader_rule = (
        "Reader-facing documents (package `TRAINING.md` and `README.md` files, "
        "`docs/*.md`, PR and issue bodies) are written for a first-time reader "
        "with no knowledge of this repository's history."
    )
    no_wrap_rule = "Do not hard-wrap markdown prose mid-sentence at a column width"
    parity_marker_rule = (
        "Keep vendor-parity and heavyweight integration tests outside regular CI "
        "and behind explicit pytest markers."
    )
    public_interface_pointer = (
        "public interfaces, and layout output schemas, including `laygen`/`posgen` "
        "shared-library work"
    )

    assert semantic_rule in core
    assert "checks raise-block and compound-suite blank lines" in core
    assert "enforce semantic, raise-block" not in core
    assert parity_marker_rule in model_skill
    assert (
        "prompt byte identity, exemplar selection, parser behavior, and "
        "repair/retry policy" in model_skill
    )
    assert public_interface_pointer in core
    assert (
        "public interface and layout output schema contracts"
        in MODEL_SKILL.read_text(encoding="utf-8").splitlines()[2]
    )
    assert "icon: lucide/...` and non-empty `tags`" in documentation_skill
    assert reader_rule not in documentation_skill
    assert no_wrap_rule not in documentation_skill
