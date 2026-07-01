#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-pupil-diameter-analysis}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda or Anaconda first, then rerun this script."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "Creating conda environment: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda activate "$ENV_NAME"
python -m pip install -e .
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate $ENV_NAME"
echo
echo "Run the CLI with:"
echo "  pupil-find --help"
echo
echo "Use this Jupyter kernel:"
echo "  Python ($ENV_NAME)"

