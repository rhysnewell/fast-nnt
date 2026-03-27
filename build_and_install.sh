#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building Python bindings (maturin develop)"
(
  cd "${repo_root}/anon-nnt-py"
  maturin develop
)

echo "==> Installing Python package (pip install .)"
(
  cd "${repo_root}/anon-nnt-py"
  pip install .
)

echo "==> Updating R bindings (rextendr::document())"
(
  cd "${repo_root}/anonnntr"
  Rscript -e "rextendr::document()"
)
