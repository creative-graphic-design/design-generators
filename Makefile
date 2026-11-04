setup:
	uv sync --all-packages --group docs --locked
	pre-commit install
