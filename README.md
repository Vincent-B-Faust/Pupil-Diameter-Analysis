# Pupil Diameter Analysis

Python tools for extracting mouse pupil boundaries from images and videos, estimating pupil diameter, and exporting quality-controlled time series.

中文说明见下方“中文快速开始”。

## Features

- Automatic eye ROI detection from dark circular seed candidates.
- Scale-adaptive thresholds for image and ROI size.
- Horizontal dark eye-fissure feature scoring.
- Corneal glint masking and inpainting before pupil segmentation.
- Least-squares circle fitting for pupil diameter.
- Automatic QC metrics and confidence score.
- Batch video processing with per-session output folders.
- Per-frame CSV export.
- Time vs. pupil diameter plot.
- Relative pupil-diameter change plot.
- Optional annotated video export.

## Repository Structure

```text
.
├── pupil_finding/
│   ├── core.py          # image-level ROI detection, pupil extraction, QC
│   ├── video.py         # video and batch processing, plotting, sessions
│   └── cli.py           # command-line interface
├── mouse_pupil_extraction_demo.ipynb
├── README_pupil_extraction_method.md
├── README_video_project.md
├── pyproject.toml
└── requirements.txt
```

## Installation

From the repository root:

```bash
python -m pip install -e .
```

If you do not want editable install:

```bash
python -m pip install -r requirements.txt
```

Run the CLI:

```bash
python -m pupil_finding.cli --help
```

After editable installation, the console script is also available:

```bash
pupil-find --help
```

## Image Analysis

```bash
pupil-find image path/to/image.jpeg \
  -o pupil_output \
  --session-name mouse01_image
```

Outputs are saved under:

```text
pupil_output/mouse01_image/
```

Image outputs:

- `<image_stem>_pupil_result.png`
- `<image_stem>_pupil_summary.csv`

## Single Video Analysis

```bash
pupil-find video path/to/video.mp4 \
  -o pupil_video_output \
  --session-name mouse01_day1 \
  --analysis-fps 30 \
  --redetect-every 30
```

Important options:

- `--analysis-fps`: frame rate used to convert frame index to seconds. If omitted, video metadata FPS is used.
- `--every-n`: analyze every Nth frame.
- `--start-frame`: first frame index to analyze.
- `--max-frames`: maximum number of analyzed frames.
- `--redetect-every`: re-detect eye ROI every N analyzed frames. Between re-detections, the previous ROI is reused.
- `--micron-per-px`: optional spatial calibration.
- `--no-annotated-video`: skip annotated video export.

Single-video outputs are saved under:

```text
pupil_video_output/<session_name>/<video_stem>/
```

Video outputs:

- `<video_stem>_pupil_timeseries.csv`
- `<video_stem>_diameter_timeseries.png`
- `<video_stem>_diameter_relative_change.png`
- `<video_stem>_annotated.mp4`
- `<video_stem>_summary.txt`

## Batch Video Analysis

Analyze all videos in a folder:

```bash
pupil-find batch path/to/video_folder \
  -o pupil_video_output \
  --session-name experiment01 \
  --analysis-fps 30
```

Analyze glob patterns:

```bash
pupil-find batch "videos/*.mp4" "videos/*.avi" \
  -o pupil_video_output \
  --session-name experiment01 \
  --every-n 2 \
  --analysis-fps 60
```

Recursive folder scan:

```bash
pupil-find batch path/to/video_folder \
  -o pupil_video_output \
  --session-name experiment01 \
  --recursive
```

Batch output structure:

```text
pupil_video_output/experiment01/
  batch_summary.csv
  video_001/
    video_001_pupil_timeseries.csv
    video_001_diameter_timeseries.png
    video_001_diameter_relative_change.png
    video_001_annotated.mp4
    video_001_summary.txt
```

## CSV Outputs

Important columns:

- `success`: whether detection succeeded for that frame.
- `frame_index`, `time_s`: frame index and timestamp.
- `roi_x`, `roi_y`, `roi_w`, `roi_h`: detected eye ROI.
- `center_x_px`, `center_y_px`: pupil center in full-frame coordinates.
- `radius_px`, `diameter_px`: fitted pupil circle measurements.
- `diameter_um`: present if `--micron-per-px` is provided.
- `diameter_baseline_px`: first successful frame diameter for that video.
- `diameter_relative_change`: fractional change from baseline.
- `diameter_relative_change_percent`: percent change from baseline.
- `qc_confidence`, `qc_confidence_level`: heuristic QC confidence score and category.
- `qc_warnings`: semicolon-separated QC warning labels.

## Algorithm Documentation

Detailed method documentation:

- [README_pupil_extraction_method.md](README_pupil_extraction_method.md)

Video/batch workflow documentation:

- [README_video_project.md](README_video_project.md)

## 中文快速开始

安装：

```bash
python -m pip install -e .
```

单个视频分析：

```bash
pupil-find video path/to/video.mp4 \
  -o pupil_video_output \
  --session-name mouse01_day1 \
  --analysis-fps 30
```

批量分析：

```bash
pupil-find batch path/to/video_folder \
  -o pupil_video_output \
  --session-name experiment01 \
  --analysis-fps 30
```

所有结果会按 session 保存，例如：

```text
pupil_video_output/experiment01/
```

每个视频会输出逐帧 CSV、时间-瞳孔直径图、相对变化率图、标注视频和 summary 文件。

## Notes

The QC confidence score is an interpretable heuristic quality score, not a probability calibrated from manual annotations. For publication-grade use, validate the workflow against manually annotated frames from representative videos.

