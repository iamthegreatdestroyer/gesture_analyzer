import os
import pandas as pd
import numpy as np

from gesture_analyzer import analyze_patterns


def make_sine_wrist(hand_index: int, frames: int = 40, cycles: int = 2):
    t = np.linspace(0, 2 * np.pi * cycles, frames)
    x = 0.5 + 0.1 * np.sin(t)  # oscillation around 0.5
    y = 0.5 + 0.05 * np.cos(t)
    rows = []
    for i in range(frames):
        rows.append([i, hand_index, "Unknown", 0, float(x[i]), float(y[i]), 0.0])
    return pd.DataFrame(rows, columns=["frame", "hand_index", "handedness", "landmark_id", "x", "y", "z"])


def test_analyze_patterns_detects_repetitions():
    df = make_sine_wrist(0, frames=60, cycles=3)
    # Add a few non-wrist landmarks to ensure grouping doesn't break
    extra = df.copy()
    extra["landmark_id"] = 1
    extra["x"] += 0.01
    all_df = pd.concat([df, extra], ignore_index=True)

    summary = analyze_patterns(all_df)
    assert summary["frames_total"] >= 60
    assert summary["frames_with_hand"] >= 60
    reps = summary["approx_repetitions_per_hand"].get("0", 0)
    # Expect at least 1 repetition due to oscillation
    assert reps >= 1
