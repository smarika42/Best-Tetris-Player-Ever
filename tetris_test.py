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
SPAWN_COLUMN = 3

try:
    import mss
except ImportError:
    mss = None

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

BOARD_COLS = 10
BOARD_ROWS = 20

PIECE_TEMPLATES = {
    'I': [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'T': [np.array([[1, 1, 1], [0, 1, 0]]), np.array([[0, 1], [1, 1], [0, 1]]), np.array([[0, 1, 0], [1, 1, 1]]), np.array([[1, 0], [1, 1], [1, 0]])],
    'S': [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])],
    'J': [np.array([[1, 0, 0], [1, 1, 1]]), np.array([[1, 1], [1, 0], [1, 0]]), np.array([[1, 1, 1], [0, 0, 1]]), np.array([[0, 1], [0, 1], [1, 1]])],
    'L': [np.array([[0, 0, 1], [1, 1, 1]]), np.array([[1, 0], [1, 0], [1, 1]]), np.array([[1, 1, 1], [1, 0, 0]]), np.array([[1, 1], [0, 1], [0, 1]])]
}

FUZZY_MATCH_THRESHOLD = 0.75

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


def find_falling_piece_component(grid, search_rows=12):
    top = grid[:search_rows].astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(top, connectivity=4)
    best_label, best_size = None, 0
    for label in range(1, num_labels):
        size = int(np.sum(labels == label))
        if 3 <= size <= 4 and size > best_size:
            best_size = size
            best_label = label
    if best_label is None:
        return None
    return (labels == best_label).astype(int)


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


# MODIFIED: Now returns the full grid and the falling mask alongside the shape!
def detect_falling_shape(frame, rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN', None, None

    grid = scan_binary_grid(crop, BOARD_COLS, BOARD_ROWS)
    mask = find_falling_piece_component(grid, search_rows=12)
    
    if mask is None:
        return 'UNKNOWN', grid, np.zeros_like(grid)

    rows_idx, cols_idx = np.where(mask == 1)
    min_r, max_r = rows_idx.min(), rows_idx.max()
    min_c, max_c = cols_idx.min(), cols_idx.max()
    piece_subgrid = mask[min_r:max_r + 1, min_c:max_c + 1]

    shape = match_shape_matrix(piece_subgrid)
    return shape, grid, mask


def detect_next_shape(frame, rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN'
    grid = scan_binary_grid(crop, 4, 4)
    return match_shape_matrix(grid)


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
        pf_rel = {"left": playfield["left"] - left, "top": playfield["top"] - top, "width": playfield["width"], "height": playfield["height"]}
        nb_rel = {"left": next_box["left"] - left, "top": next_box["top"] - top, "width": next_box["width"], "height": next_box["height"]}

    win_name = "AI Champion Controller"
    cv2.namedWindow(win_name)

    last_falling = 'UNKNOWN'
    sequencer = ActionSequencer()
    current_instructions = []

    print("\n💜 AI Vision Engine Online! Waiting for pieces...\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        # 1. Run the new vision detection
        falling_shape, full_grid, falling_mask = detect_falling_shape(frame, pf_rel)
        next_shape = detect_next_shape(frame, nb_rel)

        # 2. THE AI TRIGGER: If a valid piece just spawned!
        if falling_shape in PIECE_TEMPLATES and next_shape in PIECE_TEMPLATES:
            if falling_shape != last_falling:
                print(f"👀 Detected Spawn: {falling_shape} (Next: {next_shape})")
                
                # Subtract the falling piece to get the pure, settled board!
                settled_board = np.where(falling_mask == 1, 0, full_grid)
                
                # Feed it to your Champion AI
                best_move, _, _ = find_best_move(settled_board.tolist(), falling_shape, next_shape, CHAMPION_DNA)
                
                if best_move:
                    best_rot, best_col = best_move
                    move_offset = best_col - SPAWN_COLUMN
                    
                    # Sequence the controller inputs
                    current_instructions = sequencer.get_full_sequence(falling_shape, best_rot, move_offset)
                    print(f"🔥 AI COMMAND: {' -> '.join(current_instructions)}\n")
                
                last_falling = falling_shape
        elif falling_shape == 'UNKNOWN':
            # Reset the trigger when the piece drops and disappears
            last_falling = 'UNKNOWN'

        # 3. Draw overlays so you can play along
        cv2.rectangle(frame, (pf_rel["left"], pf_rel["top"]),
                      (pf_rel["left"] + pf_rel["width"], pf_rel["top"] + pf_rel["height"]), (0, 255, 0), 2)
        cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                      (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)
        
        # Display the AI's real-time instructions on the video feed
        if current_instructions:
            instruction_text = "DO: " + " -> ".join(current_instructions)
            cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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