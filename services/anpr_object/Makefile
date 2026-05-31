.PHONY: help install run-api test test-cov format lint clean docker-build docker-up docker-up-d docker-down docker-restart docker-logs docker-clean-volumes

help:
	@echo "Available commands:"
	@echo "  Local Development:"
	@echo "    make install           - Install dependencies"
	@echo "    make run-api           - Run FastAPI server"
	@echo "    make test              - Run tests"
	@echo "    make test-cov          - Run tests with coverage (minimum 96%)"
	@echo "    make format            - Format Python code"
	@echo "    make lint              - Run Python lint checks"
	@echo "    make clean             - Clean cache files"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build      - Build slim Docker image"
	@echo "    make docker-up         - Start with docker compose"
	@echo "    make docker-up-d       - Start with docker compose in detached mode"
	@echo "    make docker-down       - Stop docker compose"
	@echo "    make docker-restart    - Rebuild and restart"
	@echo "    make docker-logs       - View container logs"
	@echo "    make docker-clean-volumes - Remove persistent cache volumes"

install:
	uv sync --active --group dev

run-api:
	uv run --active python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv sync --active --group dev
	uv run --active --group dev python -m pytest tests/ -v

test-cov:
	uv sync --active --group dev
	uv run --active --group dev python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=95.2

format:
	uv run --active --group dev python -m ruff --version >/dev/null 2>&1 || (echo "Ruff is not installed in the project environment. Install dev dependencies with: uv sync --active --group dev" && exit 1)
	uv run --active --group dev python -m black --version >/dev/null 2>&1 || (echo "Black is not installed in the project environment. Install dev dependencies with: uv sync --active --group dev" && exit 1)
	uv run --active --group dev python -m ruff check --fix app tests
	uv run --active --group dev python -m black app tests

lint:
	uv run --active --group dev python -m ruff check app tests
	uv run --active python -m compileall app tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Docker commands
docker-build:
	docker build -f Dockerfile -t anpr-inference-service:slim --target runtime ..

docker-up:
	docker compose up

docker-up-d:
	docker compose up -d

docker-down:
	docker compose down

docker-restart:
	docker compose down
	docker compose build
	docker compose up

docker-logs:
	docker compose logs -f

docker-clean-volumes:
	@echo "Removing persistent cache volumes..."
	docker volume rm anpr_inference_service_paddle_cache anpr_inference_service_paddlex_cache anpr_inference_service_hf_cache || true
