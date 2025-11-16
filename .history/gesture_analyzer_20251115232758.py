from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from collections import defaultdict, deque

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
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None and total_frames:
        total_frames = min(total_frames, int(max_frames))

    rows: List[Tuple[int, int, str, int, float, float, float]] = []
    # Note: handedness labels captured inline per hand
    gesture_events: List[Dict] = []
    wrist_x_hist = defaultdict(lambda: deque(maxlen=30))
    wrist_dx_sign = defaultdict(lambda: 0)
    wrist_toggle_count = defaultdict(lambda: 0)

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

            if should_cancel is not None and should_cancel():
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
                    coords: Dict[int, Tuple[float, float, float]] = {}
                    for lm_id, lm in enumerate(hand_landmarks.landmark):
                        coords[lm_id] = (
                            float(lm.x),
                            float(lm.y),
                            float(lm.z),
                        )
                        rows.append(
                            (
                                frame_idx,
                                h_idx,
                                label,
                                lm_id,
                                coords[lm_id][0],
                                coords[lm_id][1],
                                coords[lm_id][2],
                            )
                        )
                    # Per-frame gesture classification
                    if is_thumbs_up(coords):
                        gesture_events.append(
                            {
                                "frame": frame_idx,
                                "hand_index": h_idx,
                                "label": "thumbs_up",
                            }
                        )
                    # Wave detection via wrist x oscillations
                    if 0 in coords:
                        wrist_x = coords[0][0]
                        hist = wrist_x_hist[h_idx]
                        prev_x = hist[-1] if len(hist) else wrist_x
                        dx = wrist_x - prev_x
                        sign = 1 if dx > 0 else (-1 if dx < 0 else 0)
                        if sign != 0 and sign != wrist_dx_sign[h_idx]:
                            wrist_toggle_count[h_idx] += 1
                            wrist_dx_sign[h_idx] = sign
                        hist.append(wrist_x)
                        if wrist_toggle_count[h_idx] >= 6 and len(hist) >= 15:
                            gesture_events.append(
                                {
                                    "frame": frame_idx,
                                    "hand_index": h_idx,
                                    "label": "wave",
                                }
                            )
                            wrist_toggle_count[h_idx] = 0
            frame_idx += 1
            if on_progress is not None:
                total = total_frames or (frame_idx + 1)
                on_progress(min(frame_idx, total), total)

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
    # Aggregate explicit gesture events and simple transcript
    label_counts: Dict[str, int] = {}
    for ev in gesture_events:
        label_counts[ev["label"]] = label_counts.get(ev["label"], 0) + 1

    transcript_tokens = [gesture_to_word(ev["label"]) for ev in gesture_events]
    transcript = " ".join([t for t in transcript_tokens if t])

    summary.update(
        {
            "gesture_events": gesture_events,
            "gesture_counts": label_counts,
            "transcript": transcript,
        }
    )
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


def is_thumbs_up(coords: Dict[int, Tuple[float, float, float]]) -> bool:
    """
    Heuristic thumbs-up:
    - Thumb tip (4) above wrist (0) (y smaller)
    - Other finger tips below their PIP joints (folded)
    Coordinates are normalized with y increasing downward.
    """
    req = [0, 4, 8, 6, 12, 10, 16, 14, 20, 18]
    if not all(k in coords for k in req):
        return False
    wrist_y = coords[0][1]
    thumb_tip_y = coords[4][1]
    thumb_up = thumb_tip_y < wrist_y - 0.03
    folded = (
        coords[8][1] > coords[6][1]
        and coords[12][1] > coords[10][1]
        and coords[16][1] > coords[14][1]
        and coords[20][1] > coords[18][1]
    )
    return bool(thumb_up and folded)


def gesture_to_word(label: str) -> str:
    mapping = {"thumbs_up": "yes", "wave": "hello"}
    return mapping.get(label, "")
