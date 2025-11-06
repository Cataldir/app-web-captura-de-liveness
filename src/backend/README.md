# Liveness Backend

Este diretório contém a API em FastAPI responsável por orquestrar o fluxo de captura e validação de liveness.

## Instalação

1. [Instale o `uv`](https://github.com/astral-sh/uv).
2. Execute `uv sync --dev` para instalar as dependências, incluindo ferramentas de desenvolvimento.

## Execução local

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Selecionando o mecanismo de liveness

Por padrão, a API utiliza um detector heurístico embutido. Para habilitar a biblioteca
[`liveness-detector`](https://pypi.org/project/liveness-detector/) (disponível apenas em Linux),
defina as variáveis de ambiente:

```bash
export LIVENESS_DETECTOR_PROVIDER=gesture
export LIVENESS_SOCKET_PATH=/tmp/liveness_socket  # opcional
```

Outras variáveis suportadas:

- `LIVENESS_LANGUAGE`: idioma das instruções (padrão `en`).
- `LIVENESS_NUM_GESTURES`: quantidade de gestos por desafio (padrão `2`).
- `LIVENESS_GESTURES`: lista separada por vírgulas com os gestos permitidos.

## Testes

```bash
uv run pytest
```

## Ferramentas de qualidade

```bash
uv run black . --check
uv run pylint app
```
