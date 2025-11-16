"""Batch CLI for processing multiple gesture videos.

Usage:
  python batch_cli.py --input-dir videos/ --pattern *.mp4 --output results.json --format json
  python batch_cli.py --input-dir videos/ --save-db --format csv --output summary.csv

Options:
  --input-dir PATH    Directory containing videos.
  --pattern GLOB      Filename pattern (default *.mp4).
  --max-frames N      Optional cap on frames per video.
  --output FILE       Output file path (json or csv). If omitted, prints summary.
  --format {json,csv} Output format (default json).
  --save-db           Persist each analysis to SQLite.
  --mapping FILE      Gesture mapping JSON override.
  --quiet             Reduce logging to warnings only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

import logging_setup  # local
import db
from gesture_analyzer import process_video

import logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch gesture analysis")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="*.mp4")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--format", choices=["json", "csv"], default="json")
    p.add_argument("--save-db", action="store_true")
    p.add_argument("--mapping", help="Override gesture mapping file")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def collect_videos(input_dir: str, pattern: str) -> List[Path]:
    paths = [
        Path(p) for p in glob.glob(str(Path(input_dir) / pattern), recursive=False)
    ]
    return [p for p in paths if p.is_file()]


def main() -> None:
    args = parse_args()
    if args.mapping:
        os.environ["GESTURE_MAPPING_PATH"] = args.mapping

    logging_setup.setup_logging()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    db.init_db()

    videos = collect_videos(args.input-dir if hasattr(args, 'input-dir') else args.input_dir, args.pattern)
    if not videos:
        print("No videos matched pattern.")
        return

    all_rows: List[Dict] = []
    csv_frames: List[pd.DataFrame] = []

    for idx, vid in enumerate(videos, start=1):
        logging.info("[%d/%d] Processing %s", idx, len(videos), vid.name)
        try:
            result = process_video(
                str(vid), max_frames=args.max_frames
            )
            summary = result.summary
            summary["video_name"] = vid.name
            summary["row_id"] = None
            if args.save_db:
                row_id = db.save_analysis(str(vid), result.dataframe, summary)
                summary["row_id"] = row_id
            all_rows.append(summary)
            csv_frames.append(result.dataframe.assign(video=vid.name))
        except Exception as e:  # noqa: BLE001
            logging.error("Failed %s: %s", vid, e)

    if args.output:
        out_path = Path(args.output)
        if args.format == "json":
            out_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
            logging.info("Wrote JSON summary: %s", out_path)
        else:  # csv
            # Flatten JSON summaries to DataFrame
            df = pd.json_normalize(all_rows)
            df.to_csv(out_path, index=False)
            logging.info("Wrote CSV summary: %s", out_path)
    else:
        print(json.dumps(all_rows, indent=2))

    # Optional raw landmarks export if CSV chosen and output provided
    if args.output and args.format == "csv":
        raw_path = Path(args.output).with_name(
            Path(args.output).stem + "_landmarks.csv"
        )
        if csv_frames:
            pd.concat(csv_frames).to_csv(raw_path, index=False)
            logging.info("Wrote landmarks CSV: %s", raw_path)


if __name__ == "__main__":  # pragma: no cover
    main()
