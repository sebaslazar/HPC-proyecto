#!/usr/bin/env bash
# Un comando: instala el entorno exacto, verifica la correctitud y regenera resultados/*.json.
#
#   bash reproducir.sh
#
# Requisito: uv (https://docs.astral.sh/uv/). Sin intervención manual.
# Si cualquier paso falla (p. ej. una prueba FAIL), el script se detiene con error.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> 1/3 Entorno exacto (uv.lock)"
uv sync --frozen

echo "==> 2/3 Correctitud"
uv run --frozen python test_correctitud.py

echo "==> 3/3 Mediciones (resultados/*.json)"
uv run --frozen python bench.py

echo "==> Listo: resultados en resultados/"
