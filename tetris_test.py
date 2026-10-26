import argparse
import json
import os
import time

import cv2
import numpy as np

from tetris_ai_search import find_best_move, BOARD_CACHE

# Clean Right-Well Tetris DNA: Keeps stack flat, never buries holes, and hunts for Tetrises
CHAMPION_DNA = {
    'lines_1': -40.0,    # Slight preference to save lines for a Tetris, but won't suicide to avoid a clear!
    'lines_2': 50.0,
    'lines_3': 200.0,
    'lines_4': 5000.0,   # Massive Tetris jackpot
    'height': 65.0,
    'holes': 900.0,      # Absolute refusal to bury holes
    'bumpiness': 150.0,  # Keeps columns 0-8 flat as a tabletop
    'wells': 140.0       # Builds a clean single well in Column 9
}

# Clean-Up DNA: Automatically activates whenever there is a hole or the stack reaches Row 13
CLEANUP_DNA = {
    'lines_1': 500.0,    # Immediately clear single/double lines to dig down and keep the board clean!
    'lines_2': 1200.0,
    'lines_3': 2200.0,
    'lines_4': 5000.0,
    'height': 85.0,
    'holes': 950.0,
    'bumpiness': 140.0,
    'wells': 50.0
}

# Exact NES Tetris leftmost bounding-box column for each piece & rotation (0=0°, 1=90°CW, 2=180°, 3=90°CCW)
NES_SPAWN_LEFT_COL = {
    'I': {0: 3, 1: 5, 2: 3, 3: 5},
    'O': {0: 4, 1: 4, 2: 4, 3: 4},
    'T': {0: 4, 1: 4, 2: 4, 3: 4},
    'S': {0: 4, 1: 5, 2: 4, 3: 5},
    'Z': {0: 4, 1: 5, 2: 4, 3: 5},
    'J': {0: 4, 1: 4, 2: 4, 3: 4},
    'L': {0: 4, 1: 4, 2: 4, 3: 4}
}

try:
    import mss
except ImportError:
    mss = None

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

BOARD_COLS = 10
BOARD_ROWS = 20

# Default NES spawn orientations
PIECE_TEMPLATES = {
    'I': [np.array([[1, 1, 1, 1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'T': [np.array([[1, 1, 1], [0, 1, 0]])],
    'S': [np.array([[0, 1, 1], [1, 1, 0]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]])],
    'J': [np.array([[1, 1, 1], [0, 0, 1]])],
    'L': [np.array([[1, 1, 1], [1, 0, 0]])]
}

FUZZY_MATCH_THRESHOLD = 0.70


class ActionSequencer:
    def get_full_sequence(self, piece_type, best_rot, target_left_col):
        if piece_type == 'UNKNOWN':
            return []
        actions = []

        # 1. Normalize rotation for symmetric pieces (O has 1 state; I, S, Z only need 0 or 1)
        if piece_type == 'O':
            best_rot = 0
        elif piece_type in ('I', 'S', 'Z'):
            best_rot = best_rot % 2

        # 2. Use ROTATE_CW (A button) and ROTATE_CCW (B button) for speed!
        if best_rot == 1:
            actions.append("ROTATE_CW")
        elif best_rot == 2:
            actions.append("ROTATE_CW")
            actions.append("ROTATE_CW")
        elif best_rot == 3:
            actions.append("ROTATE_CCW")

        # 3. Calculate exact horizontal shift based on true NES post-rotation column
        spawn_col = NES_SPAWN_LEFT_COL.get(piece_type, {}).get(best_rot, 4)
        move_offset = target_left_col - spawn_col

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
        raise RuntimeError("No calibration found. Run calibrate.py first.")
    with open(CALIBRATION_FILE) as f:
        return json.load(f)


def compute_color_distance_mask(crop):
    """
    Your proven Color-Distance detector (catches Level 0 AND Level 1 brown blocks!),
    enhanced with an anti-bloom cap (150) before blurring so bright 255 white blocks
    cannot bleed across cell boundaries!
    """
    h, w = crop.shape[:2]
    crop_f = crop.astype(np.float32)

    corner_h = max(4, int(h * 0.18))
    corner_w = max(4, int(w * 0.22))
    left_sky = crop_f[:corner_h, :corner_w].reshape(-1, 3)
    right_sky = crop_f[:corner_h, w - corner_w:].reshape(-1, 3)
    sky_pixels = np.vstack([left_sky, right_sky])

    bg_color = np.median(sky_pixels, axis=0)

    # 1. Color vector distance from the background
    color_diff = np.max(np.abs(crop_f - bg_color), axis=2)

    # 2. Red-Minus-Green contrast boost for brown Level 1 blocks against olive-green maze
    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]
    bg_b, bg_g, bg_r = bg_color[0], bg_color[1], bg_color[2]
    rg_shift = np.abs((r - g) - (bg_r - bg_g)) * 0.7

    # Cap max signal at 150 BEFORE blurring so a 255 white block doesn't bloom into neighbor cells!
    signal = np.clip(np.maximum(color_diff, rg_shift), 0, 150).astype(np.uint8)
    signal = cv2.GaussianBlur(signal, (3, 3), 0)

    sky_signal_max = float(np.percentile(signal[:corner_h, :corner_w], 92))
    cutoff = int(np.clip(sky_signal_max + 20, 32, 68))
    _, thresh = cv2.threshold(signal, cutoff, 255, cv2.THRESH_BINARY)
    return thresh


def scan_binary_grid(crop, cols, rows):
    thresh = compute_color_distance_mask(crop)
    
    h, w = crop.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    
    # 20% inner window: wide enough to never miss a falling piece block, tight enough to avoid neighbors
    half_w = max(2, int(cell_w * 0.20))
    half_h = max(2, int(cell_h * 0.20))
    grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            cx, cy = int(c * cell_w + cell_w / 2), int(r * cell_h + cell_h / 2)
            patch = thresh[max(0, cy - half_h):cy + half_h + 1, max(0, cx - half_w):cx + half_w + 1]
            if patch.size > 0 and (np.count_nonzero(patch) / patch.size) >= 0.42:
                grid[r, c] = 1
    return grid


def clean_settled_board(settled_grid):
    """Removes floating mid-air glare while keeping all grounded pieces intact."""
    cleaned = settled_grid.copy()
    rows, cols = cleaned.shape
    cleaned[:4, :] = 0

    num_labels, labels = cv2.connectedComponents(cleaned.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        component_mask = (labels == label)
        touches_floor = np.any(component_mask[rows - 1, :])
        if not touches_floor:
            cleaned[component_mask] = 0

    return cleaned


def find_falling_piece_component(grid, search_rows=4):
    top = grid[:search_rows].astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(top, connectivity=4)
    best_label, best_size = None, 0
    for label in range(1, num_labels):
        size = int(np.sum(labels == label))
        if size == 4 and size > best_size:
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


def detect_falling_shape(frame, rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    # Trim 4px off top and 3px off left/right so the cyan playfield frame never creates a fake dot in Col 9!
    crop = frame[y + 4:y + h - 1, x + 3:x + w - 3]
    if crop.size == 0:
        return 'UNKNOWN', None, None

    grid = scan_binary_grid(crop, BOARD_COLS, BOARD_ROWS)
    
    SPAWN_ROWS = 4
    mask_top = find_falling_piece_component(grid, search_rows=SPAWN_ROWS)
    
    full_mask = np.zeros_like(grid)
    if mask_top is None:
        return 'UNKNOWN', grid, full_mask

    full_mask[:SPAWN_ROWS, :] = mask_top

    rows_idx, cols_idx = np.where(mask_top == 1)
    min_r, max_r = rows_idx.min(), rows_idx.max()
    min_c, max_c = cols_idx.min(), cols_idx.max()
    piece_subgrid = mask_top[min_r:max_r + 1, min_c:max_c + 1]

    shape = match_shape_matrix(piece_subgrid)
    return shape, grid, full_mask


def detect_next_shape(frame, rect, block_size_px):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0 or block_size_px <= 0:
        return 'UNKNOWN'

    crop_f = crop.astype(np.float32)
    top_rim = crop_f[:2, :].reshape(-1, 3)
    bot_rim = crop_f[-2:, :].reshape(-1, 3)
    bg_color = np.median(np.vstack([top_rim, bot_rim]), axis=0)

    color_diff = np.max(np.abs(crop_f - bg_color), axis=2)
    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]
    rg_shift = np.abs((r - g) - (bg_color[2] - bg_color[1])) * 0.7
    signal = np.clip(np.maximum(color_diff, rg_shift), 0, 255).astype(np.uint8)
    signal = cv2.GaussianBlur(signal, (3, 3), 0)

    _, thresh = cv2.threshold(signal, 38, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(thresh > 0)
    if len(xs) < 10:
        return 'UNKNOWN'

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    tight_crop = thresh[min_y:max_y + 1, min_x:max_x + 1]
    th, tw = tight_crop.shape[:2]

    est_cols = int(np.clip(round(tw / block_size_px), 2, 4))
    est_rows = int(np.clip(round(th / block_size_px), 1, 2))

    if est_rows == 1 and est_cols == 4:
        return 'I'
    if est_rows == 2 and est_cols == 2:
        return 'O'

    subgrid = np.zeros((2, 3), dtype=int)
    cell_w = tw / 3.0
    cell_h = th / 2.0
    for r in range(2):
        for c in range(3):
            patch = tight_crop[int(r * cell_h):int((r + 1) * cell_h),
                               int(c * cell_w):int((c + 1) * cell_w)]
            if patch.size > 0 and (np.count_nonzero(patch) / patch.size) > 0.22:
                subgrid[r, c] = 1

    return match_shape_matrix(subgrid)


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

    sequencer = ActionSequencer()
    current_instructions = []
    
    candidate_shape = 'UNKNOWN'
    streak_count = 0
    empty_frames = 0
    spawn_locked = False
    last_valid_next = 'T'
    best_grid_snapshot = None

    block_size_px = pf_rel["width"] / float(BOARD_COLS)

    print("\nAI Vision Engine Online (Depth-1.5 Beam Search + Color-Distance Vision)! Waiting for pieces...\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        falling_shape, full_grid, falling_mask = detect_falling_shape(frame, pf_rel)
        next_shape = detect_next_shape(frame, nb_rel, block_size_px)
        if next_shape in PIECE_TEMPLATES:
            last_valid_next = next_shape

        raw_settled = np.where(falling_mask == 1, 0, full_grid) if full_grid is not None else None
        clean_settled = clean_settled_board(raw_settled) if raw_settled is not None else None

        # STRICT ONE-SHOT SPAWN LOCKOUT FILTER
        if falling_shape != 'UNKNOWN':
            empty_frames = 0
            best_grid_snapshot = clean_settled.copy()
            if falling_shape == candidate_shape:
                streak_count += 1
            else:
                candidate_shape = falling_shape
                streak_count = 1
        else:
            if full_grid is not None and np.sum(full_grid[:3, :]) == 0:
                empty_frames += 1
                if empty_frames >= 3:
                    spawn_locked = False
                    streak_count = 0
                    candidate_shape = 'UNKNOWN'

        # THE AI TRIGGER
        if not spawn_locked and streak_count >= 2 and candidate_shape in PIECE_TEMPLATES:
            valid_next = next_shape if next_shape in PIECE_TEMPLATES else last_valid_next
            
            # Use CLEANUP_DNA if stack reaches Row 14 or if any buried hole exists so it clears lines immediately!
            needs_cleanup = np.any(best_grid_snapshot[:14, :] == 1)
            active_dna = CLEANUP_DNA if needs_cleanup else CHAMPION_DNA
            
            t0 = time.perf_counter()
            best_move, _, _ = find_best_move(best_grid_snapshot.tolist(), candidate_shape, valid_next, active_dna)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            
            print(f"Verified Spawn: {candidate_shape} (Next: {valid_next}) [Compute: {dt_ms:.1f} ms | Cache: {len(BOARD_CACHE)}]")
            
            if best_move:
                best_rot, best_col = best_move
                current_instructions = sequencer.get_full_sequence(candidate_shape, best_rot, best_col)
                print(f"AI COMMAND: {' -> '.join(current_instructions)}\n")
            
            spawn_locked = True

        # DRAW X-RAY DEBUG OVERLAY
        eff_left = pf_rel["left"] + 3
        eff_top = pf_rel["top"] + 4
        eff_w = pf_rel["width"] - 6
        eff_h = pf_rel["height"] - 5

        cv2.rectangle(frame, (eff_left, eff_top), (eff_left + eff_w, eff_top + eff_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                      (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)

        cv2.putText(frame, f"NEXT: {last_valid_next}", (nb_rel["left"], nb_rel["top"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if full_grid is not None and clean_settled is not None:
            cell_w = eff_w / BOARD_COLS
            cell_h = eff_h / BOARD_ROWS
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    cx = int(eff_left + c * cell_w + cell_w / 2)
                    cy = int(eff_top + r * cell_h + cell_h / 2)
                    if falling_mask[r, c] == 1:
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)     # RED = Falling Piece
                    elif clean_settled[r, c] == 1:
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)     # GREEN = Grounded Settled Block
                    else:
                        cv2.circle(frame, (cx, cy), 1, (100, 100, 100), -1) # GRAY = Empty

        if current_instructions:
            instruction_text = "DO: " + " -> ".join(current_instructions)
            cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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