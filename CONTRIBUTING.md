# Contribuindo

Obrigado por ajudar a melhorar este projeto! Siga os passos abaixo para contribuir com segurança:

1. Crie uma branch a partir de `main` e descreva claramente seu objetivo.
2. Configure os ambientes do frontend (`src/frontend`) e backend (`src/backend`) seguindo o `README.md`.
3. Antes de abrir um Pull Request, execute localmente:
   - `npm run lint` e `npm run test` no frontend.
   - `uv run pytest`, `uv run black . --check` e `uv run pylint src` no backend.
4. Inclua testes que cubram suas alterações. Pull Requests sem cobertura adequada podem ser rejeitados.
5. Abra o Pull Request relacionando a issue correspondente, descrevendo o que foi feito e os impactos esperados.

Ao contribuir, você concorda com o [Código de Conduta](CODE_OF_CONDUCT.md) deste repositório, se existir, e com a licença MIT.
