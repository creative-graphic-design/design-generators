setup:
	uv sync --all-packages --group docs
	pre-commit install
