import argparse
import json
import os
import time

import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

BOARD_COLS = 10
BOARD_ROWS = 20

# Matrix definitions for all 7 Tetris pieces (in default spawn orientation)
PIECE_TEMPLATES = {
    'I': [
        np.array([[1, 1, 1, 1]]),
        np.array([[1], [1], [1], [1]])
    ],
    'O': [
        np.array([[1, 1], [1, 1]])
    ],
    'T': [
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1], [1, 1], [0, 1]]),
        np.array([[0, 1, 0], [1, 1, 1]]),
        np.array([[1, 0], [1, 1], [1, 0]])
    ],
    'S': [
        np.array([[0, 1, 1], [1, 1, 0]]),
        np.array([[1, 0], [1, 1], [0, 1]])
    ],
    'Z': [
        np.array([[1, 1, 0], [0, 1, 1]]),
        np.array([[0, 1], [1, 1], [1, 0]])
    ],
    'J': [
        np.array([[1, 0, 0], [1, 1, 1]]),
        np.array([[1, 1], [1, 0], [1, 0]]),
        np.array([[1, 1, 1], [0, 0, 1]]),
        np.array([[0, 1], [0, 1], [1, 1]])
    ],
    'L': [
        np.array([[0, 0, 1], [1, 1, 1]]),
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[1, 1, 1], [1, 0, 0]]),
        np.array([[1, 1], [0, 1], [0, 1]])
    ]
}

# Minimum overlap (intersection-over-union) required for a fuzzy match to be
# accepted when no template matches exactly. This tolerates a single
# mis-sampled cell without letting truly unrelated shapes match.
FUZZY_MATCH_THRESHOLD = 0.75


class ScreenCapture:
    def __init__(self, region):
        if mss is None:
            raise RuntimeError("mss is not installed. Run: pip install mss")
        self.sct = mss.mss()
        self.region = region

    def read(self):
        shot = self.sct.grab(self.region)
        frame = np.array(shot)
        return True, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def release(self):
        pass


class CameraCapture:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise RuntimeError("No calibration found. Run --calibrate first.")
    with open(CALIBRATION_FILE) as f:
        return json.load(f)


def scan_binary_grid(crop, cols, rows):
    """Converts an image crop to a 2D binary matrix (1 = occupied, 0 = empty).

    Uses Otsu's method to pick the light/dark split point for the *current*
    frame, so it adapts to whatever brightness/contrast the capture happens
    to have instead of relying on one fixed threshold value.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = crop.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    # Sample a patch sized relative to the cell instead of a fixed 3px
    # radius, so this keeps working if the capture resolution changes.
    half_w = max(2, int(cell_w * 0.2))
    half_h = max(2, int(cell_h * 0.2))

    grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            cx, cy = int(c * cell_w + cell_w / 2), int(r * cell_h + cell_h / 2)
            patch = thresh[max(0, cy - half_h):cy + half_h + 1, max(0, cx - half_w):cx + half_w + 1]

            if patch.size > 0 and (np.count_nonzero(patch) / patch.size) > 0.3:
                grid[r, c] = 1

    return grid


def find_falling_piece_component(grid, search_rows=12):
    """Isolates the actively falling piece via connected-component analysis.

    The previous approach took the bounding box of *every* filled cell in
    the top N rows, which breaks as soon as a settled block on the stack
    pokes up into that region (its cells get merged into the same bounding
    box as the falling piece, corrupting the shape). This instead finds
    distinct 4-cell blobs and picks the one that actually looks like a piece.
    """
    top = grid[:search_rows].astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(top, connectivity=4)

    best_label, best_size = None, 0
    for label in range(1, num_labels):
        size = int(np.sum(labels == label))
        # A standard Tetris piece is exactly 4 cells; allow 3 to tolerate a
        # cell dropped by sampling noise, but never merge multiple pieces.
        if 3 <= size <= 4 and size > best_size:
            best_size = size
            best_label = label

    if best_label is None:
        return None

    return (labels == best_label).astype(int)


def match_shape_matrix(matrix):
    """Matches a bounding box of 1s to known Tetris shape templates.

    Tries an exact match first; if none of the same-shaped templates match
    exactly (e.g. one cell was mis-sampled), falls back to the template with
    the highest intersection-over-union overlap, accepted only above
    FUZZY_MATCH_THRESHOLD so unrelated shapes are never returned.
    """
    rows = np.any(matrix, axis=1)
    cols = np.any(matrix, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 'UNKNOWN'

    cropped = matrix[np.ix_(rows, cols)]

    best_shape, best_score = 'UNKNOWN', 0.0

    for shape, templates in PIECE_TEMPLATES.items():
        for t in templates:
            if cropped.shape != t.shape:
                continue
            if np.array_equal(cropped, t):
                return shape

            intersection = np.logical_and(cropped, t).sum()
            union = np.logical_or(cropped, t).sum()
            score = intersection / union if union else 0.0
            if score > best_score:
                best_score, best_shape = score, shape

    return best_shape if best_score >= FUZZY_MATCH_THRESHOLD else 'UNKNOWN'


def detect_falling_shape(frame, rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN'

    grid = scan_binary_grid(crop, BOARD_COLS, BOARD_ROWS)

    mask = find_falling_piece_component(grid, search_rows=12)
    if mask is None:
        return 'UNKNOWN'

    rows_idx, cols_idx = np.where(mask == 1)
    min_r, max_r = rows_idx.min(), rows_idx.max()
    min_c, max_c = cols_idx.min(), cols_idx.max()
    piece_subgrid = mask[min_r:max_r + 1, min_c:max_c + 1]

    return match_shape_matrix(piece_subgrid)


def detect_next_shape(frame, rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN'

    # Preview box is typically a smaller 4x4 or 6x6 grid
    grid = scan_binary_grid(crop, 4, 4)
    return match_shape_matrix(grid)


def main(args):
    calibration = load_calibration()
    playfield = calibration["playfield"]
    next_box = calibration["next_box"]
    # CLI --source/--camera-index override whatever is stored in the
    # calibration file (previously these args were parsed but never used).
    source = args.source or calibration.get("source", "screen")

    if source == "camera":
        camera_index = args.camera_index if args.camera_index is not None else calibration.get("camera_index", 0) or 0
        capturer = CameraCapture(camera_index)
        pf_rel, nb_rel = playfield, next_box
    else:
        left = min(playfield["left"], next_box["left"])
        top = min(playfield["top"], next_box["top"])
        right = max(playfield["left"] + playfield["width"], next_box["left"] + next_box["width"])
        bottom = max(playfield["top"] + playfield["height"], next_box["top"] + next_box["height"])
        region = {"left": left, "top": top, "width": right - left, "height": bottom - top}

        capturer = ScreenCapture(region)
        pf_rel = {"left": playfield["left"] - left, "top": playfield["top"] - top, "width": playfield["width"], "height": playfield["height"]}
        nb_rel = {"left": next_box["left"] - left, "top": next_box["top"] - top, "width": next_box["width"], "height": next_box["height"]}

    win_name = "Shape-Only Tetris Detector"
    cv2.namedWindow(win_name)

    last_falling, last_next = None, None

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        falling_shape = detect_falling_shape(frame, pf_rel)
        next_shape = detect_next_shape(frame, nb_rel)

        if falling_shape != last_falling or next_shape != last_next:
            print(f"Falling piece: {falling_shape}. Next piece: {next_shape}")
            last_falling, last_next = falling_shape, next_shape

        cv2.rectangle(frame, (pf_rel["left"], pf_rel["top"]),
                      (pf_rel["left"] + pf_rel["width"], pf_rel["top"] + pf_rel["height"]), (0, 255, 0), 2)
        cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                      (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "screen"], default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    cli_args = parser.parse_args()
    main(cli_args)