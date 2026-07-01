from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime
import glob

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .core import analyze_image, failed_row, result_to_row


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv"}


def make_session_dir(base_output_dir: str | Path, session_name: str | None = None) -> Path:
    base = Path(base_output_dir)
    if session_name is None:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = base / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
    redetect_every: int = 30,
    micron_per_px: float | None = None,
    write_annotated_video: bool = True,
    output_video_fps: float | None = None,
    analysis_fps: float | None = None,
) -> pd.DataFrame:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0
    timebase_fps = analysis_fps or source_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_for_writer = output_video_fps or (source_fps / every_n if source_fps > 0 else 30.0)
    writer = None
    annotated_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
    if write_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps_for_writer, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None
            annotated_video_path = output_dir / f"{video_path.stem}_annotated.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps_for_writer, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not create annotated video writer.")

    rows: list[dict[str, Any]] = []
    last_roi: tuple[int, int, int, int] | None = None
    processed = 0
    frame_index = start_frame
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < start_frame:
            frame_index += 1
            continue
        if (frame_index - start_frame) % every_n != 0:
            frame_index += 1
            continue
        if max_frames is not None and processed >= max_frames:
            break

        time_s = frame_index / timebase_fps if timebase_fps and timebase_fps > 0 else None
        force_redetect = last_roi is None or (redetect_every > 0 and processed % redetect_every == 0)
        try:
            result = analyze_image(frame, roi=None if force_redetect else last_roi)
        except Exception:
            try:
                result = analyze_image(frame, roi=None)
            except Exception as error:
                rows.append(failed_row(frame_index, time_s, error))
                if writer is not None:
                    writer.write(_annotate_failure(frame, error))
                processed += 1
                frame_index += 1
                continue

        last_roi = result["roi"]
        row = result_to_row(result, frame_index=frame_index, time_s=time_s, micron_per_px=micron_per_px)
        rows.append(row)
        if writer is not None:
            writer.write(result["result_image"])

        processed += 1
        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()

    table = pd.DataFrame(rows)
    table = add_relative_change_columns(table, micron_per_px=micron_per_px)
    csv_path = output_dir / f"{video_path.stem}_pupil_timeseries.csv"
    table.to_csv(csv_path, index=False)
    plot_path = output_dir / f"{video_path.stem}_diameter_timeseries.png"
    plot_diameter_timeseries(table, plot_path, video_path.stem, micron_per_px=micron_per_px)
    relative_plot_path = output_dir / f"{video_path.stem}_diameter_relative_change.png"
    plot_relative_change_timeseries(table, relative_plot_path, video_path.stem)
    summary_path = output_dir / f"{video_path.stem}_summary.txt"
    _write_summary(
        summary_path,
        video_path,
        table,
        annotated_video_path if write_annotated_video else None,
        plot_path,
        relative_plot_path,
        csv_path,
    )
    return table


def process_video_batch(
    inputs: list[str | Path],
    output_dir: str | Path,
    session_name: str | None = None,
    recursive: bool = False,
    every_n: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
    redetect_every: int = 30,
    micron_per_px: float | None = None,
    write_annotated_video: bool = True,
    output_video_fps: float | None = None,
    analysis_fps: float | None = None,
) -> tuple[Path, pd.DataFrame]:
    session_dir = make_session_dir(output_dir, session_name=session_name)
    video_paths = collect_video_paths(inputs, recursive=recursive)
    if not video_paths:
        raise FileNotFoundError("No video files found for batch analysis.")

    summary_rows: list[dict[str, Any]] = []
    for video_path in video_paths:
        video_output_dir = session_dir / video_path.stem
        table = process_video(
            video_path,
            video_output_dir,
            every_n=every_n,
            start_frame=start_frame,
            max_frames=max_frames,
            redetect_every=redetect_every,
            micron_per_px=micron_per_px,
            write_annotated_video=write_annotated_video,
            output_video_fps=output_video_fps,
            analysis_fps=analysis_fps,
        )
        total = len(table)
        success = int(table["success"].sum()) if total and "success" in table else 0
        good = table[table["success"] == True] if success and "success" in table else pd.DataFrame()  # noqa: E712
        summary_rows.append(
            {
                "video": str(video_path),
                "output_dir": str(video_output_dir),
                "frames_analyzed": total,
                "successful_frames": success,
                "success_rate": success / total if total else float("nan"),
                "diameter_px_mean": good["diameter_px"].mean() if success else float("nan"),
                "diameter_px_median": good["diameter_px"].median() if success else float("nan"),
                "qc_confidence_mean": good["qc_confidence"].mean() if success else float("nan"),
            }
        )

    batch_summary = pd.DataFrame(summary_rows)
    batch_summary.to_csv(session_dir / "batch_summary.csv", index=False)
    return session_dir, batch_summary


def collect_video_paths(inputs: list[str | Path], recursive: bool = False) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            paths.extend(p for p in iterator if p.suffix.lower() in VIDEO_EXTENSIONS)
        elif any(char in str(path) for char in "*?[]"):
            paths.extend(Path(p) for p in glob.glob(str(path)) if Path(p).suffix.lower() in VIDEO_EXTENSIONS)
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def plot_diameter_timeseries(
    table: pd.DataFrame,
    plot_path: str | Path,
    title: str,
    micron_per_px: float | None = None,
) -> None:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or "success" not in table:
        return

    good = table[table["success"] == True].copy()  # noqa: E712
    if good.empty:
        return

    y_col = "diameter_um" if micron_per_px is not None and "diameter_um" in good else "diameter_px"
    y_label = "Pupil diameter (um)" if y_col == "diameter_um" else "Pupil diameter (px)"
    x_col = "time_s" if "time_s" in good and good["time_s"].notna().any() else "frame_index"
    x_label = "Time (s)" if x_col == "time_s" else "Frame index"

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(good[x_col], good[y_col], color="#1f77b4", linewidth=1.5, marker=".", markersize=3)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(f"{title}: pupil diameter over time")
    ax1.grid(True, alpha=0.3)

    if "qc_confidence" in good:
        ax2 = ax1.twinx()
        ax2.plot(good[x_col], good["qc_confidence"], color="#d62728", linewidth=1.0, alpha=0.55)
        ax2.set_ylabel("QC confidence")
        ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_relative_change_columns(
    table: pd.DataFrame,
    micron_per_px: float | None = None,
) -> pd.DataFrame:
    if table.empty or "success" not in table or "diameter_px" not in table:
        return table

    table = table.copy()
    good = table["success"] == True  # noqa: E712
    if not good.any():
        return table

    baseline_px = float(table.loc[good, "diameter_px"].iloc[0])
    table["diameter_baseline_px"] = baseline_px
    table["diameter_relative_change"] = (table["diameter_px"] - baseline_px) / baseline_px
    table["diameter_relative_change_percent"] = table["diameter_relative_change"] * 100

    if micron_per_px is not None and "diameter_um" in table:
        baseline_um = float(table.loc[good, "diameter_um"].iloc[0])
        table["diameter_baseline_um"] = baseline_um
    return table


def plot_relative_change_timeseries(
    table: pd.DataFrame,
    plot_path: str | Path,
    title: str,
) -> None:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or "success" not in table or "diameter_relative_change_percent" not in table:
        return

    good = table[table["success"] == True].copy()  # noqa: E712
    if good.empty:
        return

    x_col = "time_s" if "time_s" in good and good["time_s"].notna().any() else "frame_index"
    x_label = "Time (s)" if x_col == "time_s" else "Frame index"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
    ax.plot(
        good[x_col],
        good["diameter_relative_change_percent"],
        color="#2ca02c",
        linewidth=1.5,
        marker=".",
        markersize=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pupil diameter change from baseline (%)")
    ax.set_title(f"{title}: relative pupil diameter change")
    ax.grid(True, alpha=0.3)

    if "qc_confidence" in good:
        ax2 = ax.twinx()
        ax2.plot(good[x_col], good["qc_confidence"], color="#d62728", linewidth=1.0, alpha=0.45)
        ax2.set_ylabel("QC confidence")
        ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_failure(frame, error: Exception):
    annotated = frame.copy()
    cv2.putText(
        annotated,
        f"pupil detection failed: {type(error).__name__}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _write_summary(
    summary_path: Path,
    video_path: Path,
    table: pd.DataFrame,
    annotated_video_path: Path | None,
    plot_path: Path | None,
    relative_plot_path: Path | None,
    csv_path: Path | None,
) -> None:
    total = len(table)
    success = int(table["success"].sum()) if total and "success" in table else 0
    lines = [
        f"video: {video_path}",
        f"frames_analyzed: {total}",
        f"successful_frames: {success}",
        f"success_rate: {success / total:.4f}" if total else "success_rate: nan",
    ]
    if success and "diameter_px" in table:
        good = table[table["success"] == True]  # noqa: E712
        lines.extend(
            [
                f"diameter_px_mean: {good['diameter_px'].mean():.6f}",
                f"diameter_px_median: {good['diameter_px'].median():.6f}",
                f"qc_confidence_mean: {good['qc_confidence'].mean():.6f}",
            ]
        )
        if "diameter_baseline_px" in good:
            lines.append(f"diameter_baseline_px: {good['diameter_baseline_px'].iloc[0]:.6f}")
    if annotated_video_path is not None:
        lines.append(f"annotated_video: {annotated_video_path}")
    if plot_path is not None:
        lines.append(f"diameter_plot: {plot_path}")
    if relative_plot_path is not None:
        lines.append(f"relative_change_plot: {relative_plot_path}")
    if csv_path is not None:
        lines.append(f"timeseries_csv: {csv_path}")
    summary_path.write_text("\n".join(lines) + "\n")
