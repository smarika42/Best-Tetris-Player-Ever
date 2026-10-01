import argparse
import json
import os
import time

import cv2
import numpy as np

# ==========================================
# 1. IMPORT YOUR CHAMPION AI ENGINE
# ==========================================
from tetris_trainer import find_best_move

# Update these with your final CSV champion numbers!
CHAMPION_DNA = {
    'lines_1': -150.0,
    'lines_2': -50.0,
    'lines_3': 10.0,
    'lines_4': 5000.0,
    'height': 25.0,
    'holes': 30.0,
    'bumpiness': 2.0,
    'wells': 15.0
}

try:
    import mss
except ImportError:
    mss = None

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

BOARD_COLS = 10
BOARD_ROWS = 20

# A freshly spawned piece must have its top row at or above this row,
# and its bottom row at or above SPAWN_MAX_BOTTOM.
SPAWN_MAX_TOP = 1
SPAWN_MAX_BOTTOM = 3

PIECE_TEMPLATES = {
    'I': [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'T': [np.array([[1, 1, 1], [0, 1, 0]]), np.array([[0, 1], [1, 1], [0, 1]]),
          np.array([[0, 1, 0], [1, 1, 1]]), np.array([[1, 0], [1, 1], [1, 0]])],
    'S': [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])],
    'J': [np.array([[1, 0, 0], [1, 1, 1]]), np.array([[1, 1], [1, 0], [1, 0]]),
          np.array([[1, 1, 1], [0, 0, 1]]), np.array([[0, 1], [0, 1], [1, 1]])],
    'L': [np.array([[0, 0, 1], [1, 1, 1]]), np.array([[1, 0], [1, 0], [1, 1]]),
          np.array([[1, 1, 1], [1, 0, 0]]), np.array([[1, 1], [0, 1], [0, 1]])]
}

FUZZY_MATCH_THRESHOLD = 0.75

# Possible (rows, cols) bounding boxes of a tetromino, used for the next-piece box
NEXT_BOX_DIMS = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2)]


# ==========================================
# 2. ACTION SEQUENCER (FOR HUMAN CONTROL)
# ==========================================
class ActionSequencer:
    def get_full_sequence(self, piece_type, best_rot, move_offset):
        if piece_type == 'UNKNOWN':
            return []
        actions = []
        for _ in range(best_rot):
            actions.append("ROTATE")
        if move_offset < 0:
            for _ in range(abs(move_offset)):
                actions.append("LEFT")
        elif move_offset > 0:
            for _ in range(move_offset):
                actions.append("RIGHT")
        actions.append("DROP")
        return actions


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


# ==========================================
# 3. VISION
# ==========================================
def scan_binary_grid(crop, cols, rows):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = crop.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
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


def find_new_piece(grid):
    """Return a full-size (20x10) mask of a freshly spawned 4-cell piece, or None.

    Only accepts a connected component of exactly 4 cells that sits in the
    spawn rows. Partial pieces, stack fragments and merged blobs are ignored.
    """
    n, labels = cv2.connectedComponents(grid.astype(np.uint8), connectivity=4)
    for label in range(1, n):
        mask = labels == label
        if mask.sum() != 4:
            continue
        rows = np.where(mask)[0]
        if rows.min() <= SPAWN_MAX_TOP and rows.max() <= SPAWN_MAX_BOTTOM:
            return mask.astype(int)
    return None


def match_shape_matrix(matrix):
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
    """Returns (shape, full_grid, full_mask). Mask is always the same size as grid."""
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN', None, None

    grid = scan_binary_grid(crop, BOARD_COLS, BOARD_ROWS)
    mask = find_new_piece(grid)

    if mask is None:
        return 'UNKNOWN', grid, np.zeros_like(grid)

    r, c = np.where(mask == 1)
    shape = match_shape_matrix(mask[r.min():r.max() + 1, c.min():c.max() + 1])
    return shape, grid, mask


def detect_next_shape(frame, rect):
    """Detect the piece in the preview box without assuming a fixed 4x4 layout.

    Finds the bounding box of the lit pixels, picks the tetromino bounding-box
    size (1x4, 2x2, 2x3...) whose cells come out most square, then samples
    each cell inside that box.
    """
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN'

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pieces should be the minority of pixels; if not, polarity is inverted
    if np.count_nonzero(thresh) > thresh.size * 0.5:
        thresh = cv2.bitwise_not(thresh)

    # Drop specks of noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ys, xs = np.where(thresh > 0)
    if len(xs) < 20:
        return 'UNKNOWN'

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return 'UNKNOWN'

    best_dims, best_err = None, 1e9
    for rows, cols in NEXT_BOX_DIMS:
        cw, ch = bw / cols, bh / rows
        err = abs(cw - ch) / max(cw, ch)
        if err < best_err:
            best_err, best_dims = err, (rows, cols)
    if best_dims is None or best_err > 0.35:
        return 'UNKNOWN'

    rows, cols = best_dims
    cw, ch = bw / cols, bh / rows
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            cx = int(x0 + c * cw + cw / 2)
            cy = int(y0 + r * ch + ch / 2)
            hw, hh = max(1, int(cw * 0.25)), max(1, int(ch * 0.25))
            patch = thresh[cy - hh:cy + hh + 1, cx - hw:cx + hw + 1]
            if patch.size > 0 and np.count_nonzero(patch) / patch.size > 0.5:
                grid[r, c] = 1

    if grid.sum() != 4:
        return 'UNKNOWN'
    return match_shape_matrix(grid)


def print_grid(grid, mask=None):
    for r in range(grid.shape[0]):
        line = ""
        for c in range(grid.shape[1]):
            if mask is not None and mask[r, c]:
                line += "@"
            elif grid[r, c]:
                line += "#"
            else:
                line += "."
        print(line)
    print()


# ==========================================
# 4. MAIN LOOP
# ==========================================
def main(args):
    calibration = load_calibration()
    playfield = calibration["playfield"]
    next_box = calibration["next_box"]
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
        pf_rel = {"left": playfield["left"] - left, "top": playfield["top"] - top,
                  "width": playfield["width"], "height": playfield["height"]}
        nb_rel = {"left": next_box["left"] - left, "top": next_box["top"] - top,
                  "width": next_box["width"], "height": next_box["height"]}

    win_name = "AI Champion Controller"
    cv2.namedWindow(win_name)

    sequencer = ActionSequencer()
    current_instructions = []

    armed = True            # ready to react to the next spawn
    last_shape = 'UNKNOWN'  # shape of the last piece we reacted to
    last_next = 'UNKNOWN'   # last successfully read next-piece shape
    frame_count = 0
    full_grid, falling_mask = None, None

    print("\n💜 AI Vision Engine Online! Waiting for pieces...")
    print("Keys: q = quit, g = print the grid the vision currently sees\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue
        frame_count += 1

        falling_shape, full_grid, falling_mask = detect_falling_shape(frame, pf_rel)
        next_shape = detect_next_shape(frame, nb_rel)

        # Remember the last good next-piece reading in case a frame fails
        if next_shape in PIECE_TEMPLATES:
            last_next = next_shape

        if args.debug and frame_count % 15 == 0:
            print(f"falling={falling_shape} next={next_shape} (held={last_next}) "
                  f"last={last_shape} armed={armed}")

        # Re-arm for the next spawn:
        #  - no fresh piece visible (it dropped/merged), or
        #  - spawn rows are empty, or
        #  - a different shape appeared
        if full_grid is not None:
            if falling_shape == 'UNKNOWN' or full_grid[:2].sum() == 0:
                armed = True
            elif falling_shape != last_shape:
                armed = True

        # THE AI TRIGGER
        if armed and falling_shape in PIECE_TEMPLATES and last_next in PIECE_TEMPLATES:
            armed = False
            last_shape = falling_shape

            # Remove the falling piece so the AI sees only the settled stack
            settled_board = np.where(falling_mask == 1, 0, full_grid)

            best_move, _, _ = find_best_move(settled_board.tolist(), falling_shape, last_next, CHAMPION_DNA)

            if best_move:
                best_rot, best_col = best_move
                spawn_col = int(np.where(falling_mask == 1)[1].min())  # detected, not hard-coded
                move_offset = best_col - spawn_col

                current_instructions = sequencer.get_full_sequence(falling_shape, best_rot, move_offset)
                print(f"👀 Spawn: {falling_shape} (Next: {last_next})")
                print(f"🔥 AI COMMAND: {' -> '.join(current_instructions)}\n")

        elif falling_shape == 'UNKNOWN':
            last_shape = 'UNKNOWN'

        # Overlays
        cv2.rectangle(frame, (pf_rel["left"], pf_rel["top"]),
                      (pf_rel["left"] + pf_rel["width"], pf_rel["top"] + pf_rel["height"]), (0, 255, 0), 2)
        cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                      (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)

        if current_instructions:
            instruction_text = "DO: " + " -> ".join(current_instructions)
            cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('g') and full_grid is not None:
            print(f"--- grid (@ = falling piece) | falling={falling_shape} next={next_shape} ---")
            print_grid(full_grid, falling_mask)

    capturer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "screen"], default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="print detection state a few times per second")
    cli_args = parser.parse_args()
    main(cli_args)