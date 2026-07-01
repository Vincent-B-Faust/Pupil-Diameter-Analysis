# Mouse Pupil Finding Video Project

This project packages the mouse pupil extraction workflow into a reusable Python command-line tool for image and video analysis.

## Install

From this directory:

```bash
/opt/anaconda3/bin/python -m pip install -e .
```

The console command will then be available as:

```bash
pupil-find --help
```

You can also run it without installation:

```bash
/opt/anaconda3/bin/python -m pupil_finding.cli --help
```

## Analyze One Image

```bash
pupil-find image pupiltest.jpeg -o pupil_output --session-name test_image
```

Outputs:

- `<image_stem>_pupil_result.png`: annotated image.
- `<image_stem>_pupil_summary.csv`: one-row measurement table.

All outputs are stored under a session folder:

```text
pupil_output/<session_name>/
```

## Analyze One Video

```bash
pupil-find video path/to/video.mp4 -o pupil_video_output --session-name mouse01_day1
```

Useful options:

```bash
pupil-find video path/to/video.mp4 \
  -o pupil_video_output \
  --session-name mouse01_day1 \
  --every-n 1 \
  --redetect-every 30 \
  --analysis-fps 30 \
  --micron-per-px 3.2
```

Options:

- `--every-n`: analyze every Nth frame.
- `--start-frame`: first frame index to process.
- `--max-frames`: stop after this many analyzed frames.
- `--analysis-fps`: FPS used to convert frame index to seconds for the time axis. If omitted, video metadata FPS is used.
- `--redetect-every`: re-detect the eye ROI every N analyzed frames. Between re-detection frames, the previous ROI is reused for speed and temporal stability. If extraction fails, the tool automatically tries full-frame ROI detection again.
- `--micron-per-px`: optional pixel-to-micron calibration.
- `--no-annotated-video`: skip annotated video writing and only save CSV.
- `--output-video-fps`: override annotated video FPS.

Video outputs:

- `<video_stem>_pupil_timeseries.csv`: per-frame pupil measurements and QC.
- `<video_stem>_diameter_timeseries.png`: time-pupil diameter plot. If `--micron-per-px` is provided, the y-axis is microns; otherwise it is pixels.
- `<video_stem>_diameter_relative_change.png`: relative pupil diameter change plot, using the first successful detection in that video as baseline.
- `<video_stem>_annotated.mp4`: annotated video with ROI, pupil contour, fitted circle, diameter, and QC confidence.
- `<video_stem>_summary.txt`: success rate and basic diameter/QC summary.

Single-video outputs are saved as:

```text
pupil_video_output/<session_name>/<video_stem>/
```

If `--session-name` is not provided, a timestamped folder such as `session_20260701_142500` is created automatically.

## Batch Analyze Videos

Analyze all videos in a folder:

```bash
pupil-find batch path/to/video_folder \
  -o pupil_video_output \
  --session-name experiment01 \
  --analysis-fps 30
```

Analyze several files or glob patterns:

```bash
pupil-find batch "videos/*.mp4" "more_videos/*.avi" \
  -o pupil_video_output \
  --session-name experiment01 \
  --every-n 2 \
  --analysis-fps 60
```

Recursive folder scan:

```bash
pupil-find batch path/to/video_folder -o pupil_video_output --recursive
```

Batch output structure:

```text
pupil_video_output/<session_name>/
  batch_summary.csv
  video_001/
    video_001_pupil_timeseries.csv
    video_001_diameter_timeseries.png
    video_001_diameter_relative_change.png
    video_001_annotated.mp4
    video_001_summary.txt
  video_002/
    video_002_pupil_timeseries.csv
    video_002_diameter_timeseries.png
    video_002_diameter_relative_change.png
    video_002_annotated.mp4
    video_002_summary.txt
```

`batch_summary.csv` contains one row per video, including analyzed frame count, success rate, mean/median diameter, and mean QC confidence.

## CSV Columns

Important output columns:

- `success`: whether pupil detection succeeded for that frame.
- `frame_index`, `time_s`: frame and timestamp.
- `roi_x`, `roi_y`, `roi_w`, `roi_h`: eye ROI.
- `center_x_px`, `center_y_px`: pupil center in full-frame coordinates.
- `radius_px`, `diameter_px`: fitted circle measurements.
- `diameter_um`: present only if `--micron-per-px` is provided.
- `diameter_baseline_px`: first successful frame diameter for that video.
- `diameter_relative_change`: fractional change from baseline, `(diameter_px - baseline_px) / baseline_px`.
- `diameter_relative_change_percent`: percent change from baseline.
- `qc_confidence`, `qc_confidence_level`: heuristic quality score and level.
- `qc_warnings`: semicolon-separated QC warnings.

## Notes

The package keeps the same algorithmic steps as the notebook:

1. Full-frame eye ROI detection from dark circular seed candidates.
2. Horizontal dark fissure feature scoring.
3. ROI-level glint removal and pupil segmentation.
4. Least-squares circle fitting.
5. Automatic QC confidence scoring.

The QC confidence is a heuristic quality score, not a probability calibrated from manual labels.
