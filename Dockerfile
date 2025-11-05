# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir uv

FROM base AS development
COPY src/backend/pyproject.toml ./
RUN uv sync --dev
COPY src/backend /app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS builder
COPY src/backend/pyproject.toml ./
RUN uv sync --no-dev
COPY src/backend /app
RUN uv run python -m compileall app

FROM python:${PYTHON_VERSION}-slim AS production
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /app
COPY --from=builder /app/.venv ${VIRTUAL_ENV}
COPY --from=builder /app/app ./app
USER 1000
EXPOSE 8000
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
