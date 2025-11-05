# Liveness Backend

Este diretório contém a API em FastAPI responsável por orquestrar o fluxo de captura e validação de liveness.

## Instalação

1. [Instale o `uv`](https://github.com/astral-sh/uv).
2. Execute `uv sync --dev` para instalar as dependências, incluindo ferramentas de desenvolvimento.

## Execução local

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Testes

```bash
uv run pytest
```

## Ferramentas de qualidade

```bash
uv run black . --check
uv run pylint app
```
