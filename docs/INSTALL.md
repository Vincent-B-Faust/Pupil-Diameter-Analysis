# Installation Guide

This project should be installed into an isolated environment. Do not install it into `base` unless you intentionally want to modify your global Python environment.

## Recommended: Conda Environment

```bash
git clone https://github.com/Vincent-B-Faust/Pupil-Diameter-Analysis.git
cd Pupil-Diameter-Analysis
bash scripts/setup_conda.sh
conda activate pupil-diameter-analysis
```

This creates:

- a conda environment named `pupil-diameter-analysis`
- an editable install of this package
- a Jupyter kernel named `Python (pupil-diameter-analysis)`

Open Jupyter with:

```bash
jupyter lab
```

Then select kernel:

```text
Python (pupil-diameter-analysis)
```

## Alternative: Python venv

```bash
git clone https://github.com/Vincent-B-Faust/Pupil-Diameter-Analysis.git
cd Pupil-Diameter-Analysis
bash scripts/setup_venv.sh
source .venv/bin/activate
```

This creates:

- a local `.venv/`
- an editable install of this package
- a Jupyter kernel named `Python (pupil-diameter-analysis venv)`

## Quick Test

```bash
pupil-find image pupiltest.jpeg -o pupil_output --session-name test_image
```

Expected output:

```text
pupil_output/test_image/
  pupiltest_pupil_result.png
  pupiltest_pupil_summary.csv
```

## CLI Usage

Single video:

```bash
pupil-find video path/to/video.mp4 \
  -o pupil_video_output \
  --session-name mouse01_day1 \
  --analysis-fps 30
```

Batch videos:

```bash
pupil-find batch path/to/video_folder \
  -o pupil_video_output \
  --session-name experiment01 \
  --analysis-fps 30
```

## Troubleshooting

If `pupil-find` is not found, activate the environment first:

```bash
conda activate pupil-diameter-analysis
```

or:

```bash
source .venv/bin/activate
```

If Jupyter does not show the environment, rerun:

```bash
python -m ipykernel install --user --name pupil-diameter-analysis --display-name "Python (pupil-diameter-analysis)"
```

If OpenCV installation fails on a managed machine, prefer the conda workflow because `conda-forge` handles OpenCV binary dependencies more reliably.

