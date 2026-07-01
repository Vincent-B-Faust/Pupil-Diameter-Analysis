from __future__ import annotations

import argparse
from pathlib import Path

from .core import save_image_outputs
from .video import make_session_dir, process_video, process_video_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pupil-find",
        description="Mouse pupil extraction for images and videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="Analyze one image.")
    image.add_argument("image_path")
    image.add_argument("-o", "--output-dir", default="pupil_output")
    image.add_argument("--session-name", default=None)
    image.add_argument("--micron-per-px", type=float, default=None)

    video = subparsers.add_parser("video", help="Analyze a video frame by frame.")
    video.add_argument("video_path")
    video.add_argument("-o", "--output-dir", default="pupil_video_output")
    video.add_argument("--session-name", default=None)
    video.add_argument("--every-n", type=int, default=1, help="Analyze every Nth frame.")
    video.add_argument("--start-frame", type=int, default=0)
    video.add_argument("--max-frames", type=int, default=None)
    video.add_argument(
        "--redetect-every",
        type=int,
        default=30,
        help="Re-detect eye ROI every N analyzed frames. Use 0 to reuse ROI until failure.",
    )
    video.add_argument("--micron-per-px", type=float, default=None)
    video.add_argument(
        "--analysis-fps",
        type=float,
        default=None,
        help="FPS used to convert frame index to time. Defaults to video metadata FPS.",
    )
    video.add_argument("--no-annotated-video", action="store_true")
    video.add_argument("--output-video-fps", type=float, default=None)

    batch = subparsers.add_parser("batch", help="Batch analyze video files.")
    batch.add_argument(
        "inputs",
        nargs="+",
        help="Video files, directories, or glob patterns. Directories scan common video extensions.",
    )
    batch.add_argument("-o", "--output-dir", default="pupil_video_output")
    batch.add_argument("--session-name", default=None)
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--every-n", type=int, default=1)
    batch.add_argument("--start-frame", type=int, default=0)
    batch.add_argument("--max-frames", type=int, default=None)
    batch.add_argument("--redetect-every", type=int, default=30)
    batch.add_argument("--micron-per-px", type=float, default=None)
    batch.add_argument(
        "--analysis-fps",
        type=float,
        default=None,
        help="FPS used to convert frame index to time. Defaults to each video's metadata FPS.",
    )
    batch.add_argument("--no-annotated-video", action="store_true")
    batch.add_argument("--output-video-fps", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "image":
        session_dir = make_session_dir(args.output_dir, args.session_name)
        table = save_image_outputs(args.image_path, session_dir, micron_per_px=args.micron_per_px)
        print(table.round(4).to_string(index=False))
        print(f"Saved image outputs to session: {session_dir.resolve()}")
        return 0

    if args.command == "video":
        session_dir = make_session_dir(args.output_dir, args.session_name)
        video_output_dir = session_dir / Path(args.video_path).stem
        table = process_video(
            args.video_path,
            video_output_dir,
            every_n=args.every_n,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            redetect_every=args.redetect_every,
            micron_per_px=args.micron_per_px,
            write_annotated_video=not args.no_annotated_video,
            output_video_fps=args.output_video_fps,
            analysis_fps=args.analysis_fps,
        )
        total = len(table)
        success = int(table["success"].sum()) if total else 0
        print(f"Analyzed frames: {total}")
        print(f"Successful frames: {success}")
        print(f"Saved video outputs to session: {session_dir.resolve()}")
        return 0

    if args.command == "batch":
        session_dir, summary = process_video_batch(
            args.inputs,
            args.output_dir,
            session_name=args.session_name,
            recursive=args.recursive,
            every_n=args.every_n,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            redetect_every=args.redetect_every,
            micron_per_px=args.micron_per_px,
            write_annotated_video=not args.no_annotated_video,
            output_video_fps=args.output_video_fps,
            analysis_fps=args.analysis_fps,
        )
        print(summary.round(4).to_string(index=False))
        print(f"Saved batch outputs to session: {session_dir.resolve()}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
