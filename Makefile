.PHONY: install format lint test

install:
	@if uv --version >/dev/null 2>&1; then \
		uv sync --all-groups; \
		uv run pre-commit install; \
	else \
		pip install -e ".[dev]"; \
		pre-commit install; \
	fi

format:
	@if uv --version >/dev/null 2>&1; then \
		uv run ruff format tcaf_model; \
		uv run ruff check --fix tcaf_model; \
	else \
		ruff format tcaf_model; \
		ruff check --fix tcaf_model; \
	fi

lint:
	@if uv --version >/dev/null 2>&1; then \
		uv run ruff check tcaf_model; \
	else \
		ruff check tcaf_model; \
	fi

test:
	@if uv --version >/dev/null 2>&1; then \
		uv run pytest; \
	else \
		pytest; \
	fi
