from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands


@dataclass
class ProcessResult:
    dataframe: pd.DataFrame
    summary: Dict


def process_video(
    video_path: str,
    max_frames: int | None = None,
) -> ProcessResult:
    """
    Process a video to extract MediaPipe Hand landmarks per frame.

    Returns a DataFrame with columns:
    [frame, hand_index, handedness, landmark_id, x, y, z]
    and a simple summary dict of pattern metrics.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows: List[Tuple[int, int, str, int, float, float, float]] = []
    # Note: handedness labels captured inline per hand

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                # handedness classification (Left/Right) if available
                labels = []
                if result.multi_handedness:
                    labels = [
                        h.classification[0].label
                        for h in result.multi_handedness
                    ]

                for h_idx, hand_landmarks in enumerate(
                    result.multi_hand_landmarks
                ):
                    label = labels[h_idx] if h_idx < len(labels) else "Unknown"
                    for lm_id, lm in enumerate(hand_landmarks.landmark):
                        rows.append(
                            (
                                frame_idx,
                                h_idx,
                                label,
                                lm_id,
                                float(lm.x),
                                float(lm.y),
                                float(lm.z),
                            )
                        )
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "hand_index",
            "handedness",
            "landmark_id",
            "x",
            "y",
            "z",
        ],
    )

    summary = analyze_patterns(df)
    return ProcessResult(dataframe=df, summary=summary)


def analyze_patterns(df: pd.DataFrame) -> Dict:
    """
    Very simple pattern analysis to bootstrap iteration:
    - frames_total: total frames observed
    - frames_with_hand: frames containing any hand landmarks
        - approx_repetitions_per_hand: estimate repetitions from wrist (lm 0)
            x/y zero-crossings
    - avg_motion_per_hand: average per-frame wrist displacement
    """
    if df.empty:
        return {
            "frames_total": 0,
            "frames_with_hand": 0,
            "approx_repetitions_per_hand": {},
            "avg_motion_per_hand": {},
        }

    frames_total = int(df["frame"].max()) + 1
    frames_with_hand = int(df.groupby("frame").size().shape[0])

    summary: Dict = {
        "frames_total": frames_total,
        "frames_with_hand": frames_with_hand,
        "approx_repetitions_per_hand": {},
        "avg_motion_per_hand": {},
    }

    # Use wrist (landmark 0) trajectory per hand
    for hand_idx, g in df[df["landmark_id"] == 0].groupby("hand_index"):
        g_sorted = g.sort_values("frame")
        x = g_sorted["x"].to_numpy()
        y = g_sorted["y"].to_numpy()
        if x.size < 3:
            summary["approx_repetitions_per_hand"][str(hand_idx)] = 0
            summary["avg_motion_per_hand"][str(hand_idx)] = 0.0
            continue

        # Displacement and simple speed proxy
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2)
        avg_speed = float(np.mean(speed))

        # Approximate repetitions by zero-crossings in velocity (sign changes)
        # Count sign changes where magnitude is above a small threshold
        sign_changes = np.sum(np.abs(np.diff(np.sign(dx))) > 0)
        sign_changes += np.sum(np.abs(np.diff(np.sign(dy))) > 0)
        # Scale down as both axes contribute
        approx_reps = int(max(0, round(sign_changes / 4)))

        summary["approx_repetitions_per_hand"][str(hand_idx)] = approx_reps
        summary["avg_motion_per_hand"][str(hand_idx)] = avg_speed

    return summary
