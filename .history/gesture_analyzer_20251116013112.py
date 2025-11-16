from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from collections import defaultdict, deque

import cv2
try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None
import numpy as np
import pandas as pd
import json
import os

mp_hands = mp.solutions.hands if mp else None

_MAPPING_CACHE: Dict[str, str] | None = None


def load_gesture_mapping() -> Dict[str, str]:
    """Load gesture->word mapping from JSON file or return defaults.

    Environment variable GESTURE_MAPPING_PATH can override filename.
    File format: {"gesture_label": "word"}
    """
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        return _MAPPING_CACHE
    filename = os.environ.get("GESTURE_MAPPING_PATH", "gesture_mapping.json")
    path = Path(filename)
    if path.exists():
        try:
            data = json.load(path.open("r", encoding="utf-8"))
            if isinstance(data, dict):
                _MAPPING_CACHE = {k: str(v) for k, v in data.items()}
                return _MAPPING_CACHE
        except Exception:
            pass  # fall back to defaults
    _MAPPING_CACHE = {
        "thumbs_up": "yes",
        "wave": "hello",
        "ok": "ok",
        "pinch": "select",
        "point": "point",
    }
    return _MAPPING_CACHE


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
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    rows: List[Tuple[int, int, str, int, float, float, float]] = []
    # Note: handedness labels captured inline per hand
    gesture_events: List[Dict] = []
    wrist_x_hist = defaultdict(lambda: deque(maxlen=30))
    wrist_dx_sign = defaultdict(lambda: 0)
    wrist_toggle_count = defaultdict(lambda: 0)

    if mp_hands is None:
        # Return empty result with mediapipe unavailable note
        empty_df = pd.DataFrame(
            columns=["frame", "hand_index", "handedness", "landmark_id", "x", "y", "z"]
        )
        summary = {
            "frames_total": 0,
            "frames_with_hand": 0,
            "approx_repetitions_per_hand": {},
            "avg_motion_per_hand": {},
            "gesture_events": [],
            "gesture_counts": {},
            "transcript": "",
            "transcript_decoded": "",
            "transcript_sentences": [],
            "mediapipe_unavailable": True,
        }
        return ProcessResult(dataframe=empty_df, summary=summary)

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
                    if is_thumbs_up(coords, label):
                        gesture_events.append(
                            {
                                "frame": frame_idx,
                                "hand_index": h_idx,
                                "label": "thumbs_up",
                            }
                        )
                    if is_ok_sign(coords):
                        gesture_events.append(
                            {
                                "frame": frame_idx,
                                "hand_index": h_idx,
                                "label": "ok",
                            }
                        )
                    if is_pinch(coords):
                        gesture_events.append(
                            {
                                "frame": frame_idx,
                                "hand_index": h_idx,
                                "label": "pinch",
                            }
                        )
                    if is_point(coords):
                        gesture_events.append(
                            {
                                "frame": frame_idx,
                                "hand_index": h_idx,
                                "label": "point",
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
    decoded = decode_gesture_sequence(gesture_events, fps=fps)

    summary.update(
        {
            "gesture_events": gesture_events,
            "gesture_counts": label_counts,
            "transcript": transcript,
            "transcript_decoded": decoded.get("transcript", ""),
            "transcript_sentences": decoded.get("sentences", []),
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


def is_thumbs_up(
    coords: Dict[int, Tuple[float, float, float]], handedness: str
) -> bool:
    """Robust thumbs-up heuristic using angles and handedness side test.

    Conditions:
    - Thumb tip (4) above wrist (0) by margin (0.03)
    - Other finger tips folded (tip y > PIP y because y increases downward)
    - Angle at thumb MCP (landmarks 1-2-3) > 150 degrees (extended)
        - Side position:
            * Right hand: thumb tip x < index MCP (5) x
            * Left hand:  thumb tip x > index MCP x
    """
    needed = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
    if not all(k in coords for k in needed):
        return False
    wrist_y = coords[0][1]
    thumb_tip_y = coords[4][1]
    thumb_up = thumb_tip_y < wrist_y - 0.03
    angle_thumb = compute_angle(coords[1], coords[2], coords[3])
    thumb_extended = angle_thumb > 150.0
    folded = (
        coords[8][1] > coords[6][1]
        and coords[12][1] > coords[10][1]
        and coords[16][1] > coords[14][1]
        and coords[20][1] > coords[18][1]
    )
    side_ok = True
    index_mcp_x = coords[5][0]
    thumb_tip_x = coords[4][0]
    hl = handedness.lower()
    if hl == "right":
        side_ok = thumb_tip_x < index_mcp_x - 0.01
    elif hl == "left":
        side_ok = thumb_tip_x > index_mcp_x + 0.01
    return bool(thumb_up and thumb_extended and folded and side_ok)


def gesture_to_word(label: str) -> str:
    mapping = load_gesture_mapping()
    return mapping.get(label, "")


def compute_angle(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
) -> float:
    """Return angle at p2 (degrees) for segments p2->p1 and p2->p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def decode_gesture_sequence(
    events: List[Dict],
    fps: float,
    word_gap_s: float = 0.9,
    sentence_gap_s: float = 1.8,
) -> Dict:
    """Convert gesture events into a punctuated transcript using timing gaps.

    - If time gap > sentence_gap_s: insert period.
    - Else if gap > word_gap_s: insert space (new word).
    - Duplicate same-label events within 0.3s are collapsed.
    Returns dict with 'transcript' and 'sentences'.
    """
    if not events:
        return {"transcript": "", "sentences": []}
    sorted_events = sorted(events, key=lambda e: e["frame"])
    fps_eff = fps if fps and fps > 0 else 30.0
    transcript_parts: List[str] = []
    last_frame = None
    last_token = None
    for ev in sorted_events:
        token = gesture_to_word(ev["label"])
        if not token:
            continue
        if last_frame is None:
            transcript_parts.append(token)
            last_frame = ev["frame"]
            last_token = token
            continue
        dt = (ev["frame"] - last_frame) / fps_eff
        if token == last_token and dt < 0.3:
            last_frame = ev["frame"]
            continue
        if dt > sentence_gap_s:
            transcript_parts.append(".")
        elif dt > word_gap_s:
            transcript_parts.append(" ")
        else:
            transcript_parts.append(" ")
        transcript_parts.append(token)
        last_frame = ev["frame"]
        last_token = token
    raw = "".join(transcript_parts).strip()
    if raw and not raw.endswith("."):
        raw += "."
    sentences = [s.strip() for s in raw.split(".") if s.strip()]
    return {"transcript": raw, "sentences": sentences}


# --- Additional gesture heuristics ---

def distance(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    return float(
        np.linalg.norm(
            np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        )
    )


def is_ok_sign(coords: Dict[int, Tuple[float, float, float]]) -> bool:
    """OK sign: thumb tip touches index tip; other three fingers extended."""
    req = [4, 8, 12, 10, 16, 14, 20, 18]
    if not all(k in coords for k in req):
        return False
    if distance(coords[4], coords[8]) > 0.05:
        return False
    extended = (
        coords[12][1] < coords[10][1]
        and coords[16][1] < coords[14][1]
        and coords[20][1] < coords[18][1]
    )
    return bool(extended)


def is_pinch(coords: Dict[int, Tuple[float, float, float]]) -> bool:
    """Pinch: thumb tip near index tip; middle finger folded."""
    req = [4, 8, 12, 10]
    if not all(k in coords for k in req):
        return False
    if distance(coords[4], coords[8]) > 0.035:
        return False
    middle_folded = coords[12][1] > coords[10][1]
    return bool(middle_folded)


def is_point(coords: Dict[int, Tuple[float, float, float]]) -> bool:
    """Point: index extended; others folded; thumb near index MCP."""
    req = [4, 5, 8, 6, 12, 10, 16, 14, 20, 18]
    if not all(k in coords for k in req):
        return False
    index_extended = coords[8][1] < coords[6][1]
    others_folded = (
        coords[12][1] > coords[10][1]
        and coords[16][1] > coords[14][1]
        and coords[20][1] > coords[18][1]
    )
    thumb_near_index_mcp = distance(coords[4], coords[5]) < 0.08
    return bool(index_extended and others_folded and thumb_near_index_mcp)
