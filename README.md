# App Web Captura de Liveness

Esta aplicação demonstra uma solução de captura de liveness com um frontend em Next.js e um backend em FastAPI capaz de receber streaming de vídeo via WebSocket.

## Estrutura do repositório

```text
src/
  frontend/   # Next.js 14 (App Router), TailwindCSS e TypeScript
  backend/    # FastAPI com suporte a WebSocket e validações de liveness
```

Outros arquivos importantes:

- `Dockerfile`: build multi-stage com imagens de desenvolvimento e produção.
- `.github/workflows/ci.yml`: pipeline que executa lint e testes em ambos os componentes.
- `CONTRIBUTING.md`: guia para contribuir de forma segura.

## Pré-requisitos

- Node.js 20+
- `uv` e Python 3.13+
- Docker (opcional, para execução containerizada)

## Frontend (Next.js)

```bash
cd src/frontend
npm install
npm run dev
```

A página única (`/`) captura a câmera do usuário, envia os chunks do `MediaRecorder` pelo WebSocket (`ws://localhost:8000/ws/liveness` por padrão) e exibe o estado retornado pelo backend. Configure `NEXT_PUBLIC_BACKEND_WS_URL` caso utilize outra URL.

Testes e lint:

```bash
npm run lint
npm run test
```

## Backend (FastAPI + uv)

```bash
cd src/backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /`: health probe para verificação de prontidão.
- `POST /validate`: recebe um payload com amostras base64 e retorna o resultado agregado de liveness.
- `WS /ws/liveness`: recebe frames binários do frontend e responde com mensagens de validação em tempo real.

Testes e qualidade:

```bash
uv run pytest
uv run black . --check
uv run pylint app
```

## Docker

```bash
docker build -t liveness-backend:dev --target development .
docker build -t liveness-backend:prod --target production .
```

A build `development` mantém as dependências de desenvolvimento e ativo o reload. A build `production` copia apenas o necessário para executar a API em um ambiente enxuto.

## Contribuindo

Siga as instruções descritas em `CONTRIBUTING.md` e observe o `CODE_OF_CONDUCT.md`.
