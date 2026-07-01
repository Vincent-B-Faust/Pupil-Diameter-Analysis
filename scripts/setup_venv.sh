#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN=/path/to/python or install Python 3.9+."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install ipykernel jupyterlab
python -m ipykernel install --user --name "pupil-diameter-analysis-venv" --display-name "Python (pupil-diameter-analysis venv)"

echo
echo "Virtual environment is ready."
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Run the CLI with:"
echo "  pupil-find --help"
echo
echo "Use this Jupyter kernel:"
echo "  Python (pupil-diameter-analysis venv)"

