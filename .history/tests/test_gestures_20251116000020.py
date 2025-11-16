from gesture_analyzer import (
    is_thumbs_up,
    decode_gesture_sequence,
)

# Synthetic coordinate helper
# Each landmark: (x, y, z) with y increasing downward.

def make_thumbsup_coords(handedness="right"):
    # Provide required landmarks for thumbs-up detection
    coords = {}
    # Wrist
    coords[0] = (0.5, 0.6, 0.0)
    # Thumb CMC->MCP->IP roughly horizontal line, tip above wrist
    coords[1] = (0.45, 0.6, 0.0)
    coords[2] = (0.47, 0.6, 0.0)
    coords[3] = (0.49, 0.6, 0.0)
    coords[4] = (0.48, 0.55, 0.0)  # tip above
    # Index MCP/PIP/DIP/TIP folded (tip y > PIP y)
    coords[5] = (0.55, 0.6, 0.0)
    coords[6] = (0.55, 0.58, 0.0)
    coords[7] = (0.55, 0.59, 0.0)
    coords[8] = (0.55, 0.62, 0.0)
    # Middle
    coords[9] = (0.57, 0.6, 0.0)
    coords[10] = (0.57, 0.58, 0.0)
    coords[11] = (0.57, 0.59, 0.0)
    coords[12] = (0.57, 0.62, 0.0)
    # Ring
    coords[13] = (0.59, 0.6, 0.0)
    coords[14] = (0.59, 0.58, 0.0)
    coords[15] = (0.59, 0.59, 0.0)
    coords[16] = (0.59, 0.62, 0.0)
    # Pinky
    coords[17] = (0.61, 0.6, 0.0)
    coords[18] = (0.61, 0.58, 0.0)
    coords[19] = (0.61, 0.59, 0.0)
    coords[20] = (0.61, 0.62, 0.0)
    return coords


def test_is_thumbs_up_right():
    coords = make_thumbsup_coords("right")
    assert is_thumbs_up(coords, "right") is True


def test_decode_gesture_sequence_basic():
    # Create events spaced to form words and sentences
    # fps = 30, sentence gap > 1.8s -> >54 frames
    events = [
        {"frame": 0, "label": "thumbs_up"},
        {"frame": 40, "label": "wave"},  # word gap < sentence gap
        {"frame": 110, "label": "thumbs_up"},  # sentence boundary
    ]
    result = decode_gesture_sequence(events, fps=30.0)
    transcript = result["transcript"]
    assert "yes" in transcript and "hello" in transcript
    # Should end with period
    assert transcript.endswith(".")
