from pixel_vae.testing import JSONValue


def _should_stop_after_divergence(
    first_divergence: dict[str, JSONValue] | None, record_only: bool
) -> bool:
    return first_divergence is not None and not record_only


def _retain_first_divergence(
    first_divergence: dict[str, JSONValue] | None,
    current_record: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    return first_divergence if first_divergence is not None else current_record
