.PHONY: run lint lint-fix format format-check test test-cov sync help

run:
	uv run main

lint:
	uv run ruff check app modules triton_server tests

lint-fix:
	uv run ruff check --fix app modules triton_server tests

format:
	uv run black app modules triton_server tests

format-check:
	uv run black --check app modules triton_server tests

test:
	uv run pytest tests

test-cov:
	uv run pytest --cov=app --cov=modules --cov=triton_server --cov-report=term-missing --cov-report=term tests

sync:
	uv sync

help:
	@echo "Available commands:"
	@echo "  make run       - Start the API server"
	@echo "  make lint      - Run ruff linting"
	@echo "  make lint-fix  - Auto-fix ruff linting issues"
	@echo "  make format    - Format code with black"
	@echo "  make format-check - Check code formatting"
	@echo "  make test      - Run pytest tests"
	@echo "  make test-cov  - Run pytest with coverage"
	@echo "  make sync      - Sync dependencies with uv"
	@echo "  make help      - Show this help message"
