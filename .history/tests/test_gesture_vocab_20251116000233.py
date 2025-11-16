import json
from pathlib import Path

import gesture_analyzer as ga


def test_mapping_override(tmp_path, monkeypatch):
    # Prepare custom mapping file
    custom = {"thumbs_up": "affirm", "wave": "hi"}
    mapping_path = tmp_path / "map.json"
    mapping_path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setenv("GESTURE_MAPPING_PATH", str(mapping_path))
    ga._MAPPING_CACHE = None

    loaded = ga.load_gesture_mapping()
    assert loaded["thumbs_up"] == "affirm"
    assert ga.gesture_to_word("wave") == "hi"


def test_ok_pinch_point_detection():
    # Synthetic minimal coords for ok, pinch, point
    # Provide required landmarks with rough plausible layout
    # OK sign: thumb tip near index tip, other fingers extended
    ok_coords = {
        4: (0.4, 0.5, 0.0),  # thumb tip
        8: (0.405, 0.505, 0.0),  # index tip
        12: (0.45, 0.48, 0.0),  # middle tip (extended)
        10: (0.45, 0.50, 0.0),
        16: (0.47, 0.48, 0.0),  # ring tip
        14: (0.47, 0.50, 0.0),
        20: (0.49, 0.48, 0.0),  # pinky tip
        18: (0.49, 0.50, 0.0),
    }
    assert ga.is_ok_sign(ok_coords) is True

    # Pinch: thumb near index tip, middle folded
    pinch_coords = {
        4: (0.4, 0.5, 0.0),
        8: (0.402, 0.502, 0.0),
        12: (0.45, 0.52, 0.0),  # middle tip folded (y greater)
        10: (0.45, 0.50, 0.0),
    }
    assert ga.is_pinch(pinch_coords) is True

    # Point: index extended, others folded, thumb near index MCP (5)
    point_coords = {
        4: (0.4, 0.52, 0.0),  # thumb tip
        5: (0.42, 0.55, 0.0),  # index MCP
        8: (0.42, 0.50, 0.0),  # index tip extended (y less than PIP)
        6: (0.42, 0.52, 0.0),
        12: (0.46, 0.57, 0.0),  # middle tip folded
        10: (0.46, 0.55, 0.0),
        16: (0.48, 0.57, 0.0),  # ring tip folded
        14: (0.48, 0.55, 0.0),
        20: (0.50, 0.57, 0.0),  # pinky tip folded
        18: (0.50, 0.55, 0.0),
    }
    assert ga.is_point(point_coords) is True
