import argparse
import json
import os
import time

import cv2
import numpy as np

from tetris_ai_search import find_best_move, BOARD_CACHE

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# Tuned to aggressively build a flat 9-column stack and cash in 4-line Tetrises in Column 9!
CHAMPION_DNA = {
    'lines_1': -180.0,   # Discourage burning single lines when stack is safe
    'lines_2': -60.0,    # Discourage burning doubles when stack is safe
    'lines_3': 100.0,
    'lines_4': 8000.0,   # MASSIVE 4-LINE TETRIS PRIORITY!
    'height': 55.0,
    'holes': 1100.0,     # Never bury holes
    'bumpiness': 185.0,  # Keep Columns 0..8 ultra-flat
    'wells': 260.0       # Guard Column 9 religiously for the 'I' piece
}

CLEANUP_DNA = {
    'lines_1': 650.0,
    'lines_2': 1400.0,
    'lines_3': 2600.0,
    'lines_4': 8000.0,
    'height': 95.0,
    'holes': 1050.0,
    'bumpiness': 160.0,
    'wells': 60.0
}

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

PIECE_TEMPLATES = {
    'I': np.array([[1, 1, 1, 1]], dtype=np.float32),
    'O': np.array([[1, 1], [1, 1]], dtype=np.float32),
    'T': np.array([[1, 1, 1], [0, 1, 0]], dtype=np.float32),
    'S': np.array([[0, 1, 1], [1, 1, 0]], dtype=np.float32),
    'Z': np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32),
    'J': np.array([[1, 1, 1], [0, 0, 1]], dtype=np.float32),
    'L': np.array([[1, 1, 1], [1, 0, 0]], dtype=np.float32)
}

SMOOTH_BG_COLOR = None


class ActionSequencer:
    def get_full_sequence(self, piece_type, best_rot, target_left_col):
        if piece_type == 'UNKNOWN':
            return []
        actions = []

        if piece_type == 'O':
            best_rot = 0
        elif piece_type in ('I', 'S', 'Z'):
            best_rot = best_rot % 2

        if best_rot == 1:
            actions.append("ROTATE_CW")
        elif best_rot == 2:
            actions.append("ROTATE_CW")
            actions.append("ROTATE_CW")
        elif best_rot == 3:
            actions.append("ROTATE_CCW")

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
    """
    Hybrid Camera Driver:
    - Automatically uses Picamera2 (libcamera) on Raspberry Pi for the Arducam UC-439 (OV5647).
    - Settles auto-exposure on the TV monitor and locks AE/AWB so colors never shift mid-game!
    - Falls back to standard cv2.VideoCapture if run on a laptop/desktop with a USB webcam.
    """
    def __init__(self, index=0, width=640, height=480):
        self.use_picam = HAS_PICAMERA2
        self.picam2 = None
        self.cap = None

        if self.use_picam:
            try:
                print("🍓 Initializing Arducam OV5647 via Picamera2 (640x480)...")
                self.picam2 = Picamera2(camera_num=index)
                config = self.picam2.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"},
                    controls={"FrameDurationLimits": (16666, 33333)}  # 30-60 FPS
                )
                self.picam2.configure(config)
                self.picam2.start()

                # Let auto-exposure and white balance settle on the monitor for 0.8s, then lock them!
                time.sleep(0.8)
                self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})
                print("🔒 OV5647 Exposure & White Balance locked for consistent piece colors!")
                return
            except Exception as e:
                print(f"⚠️ Picamera2 init failed ({e}), falling back to cv2.VideoCapture({index})...")
                self.use_picam = False

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self.use_picam and self.picam2 is not None:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            # Picamera2's "RGB888" format outputs BGR byte order directly compatible with OpenCV!
            return True, frame
        return self.cap.read()

    def release(self):
        if self.use_picam and self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
        elif self.cap is not None:
            self.cap.release()


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise RuntimeError("No calibration found. Run calibrate.py first.")
    with open(CALIBRATION_FILE) as f:
        return json.load(f)


def get_trimmed_playfield_rect(rect):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    trim_x = max(1, int(round(w * 0.02)))
    trim_top = max(1, int(round(h * 0.015)))
    trim_bot = max(1, int(round(h * 0.015)))
    return {
        "left": x + trim_x,
        "top": y + trim_top,
        "width": max(10, w - 2 * trim_x),
        "height": max(20, h - trim_top - trim_bot)
    }


def compute_cell_signal_map(crop):
    global SMOOTH_BG_COLOR
    h, w = crop.shape[:2]
    crop_f = crop.astype(np.float32)

    corner_h = max(4, int(h * 0.14))
    corner_w = max(4, int(w * 0.25))
    left_sky = crop_f[:corner_h, :corner_w].reshape(-1, 3)
    right_sky = crop_f[:corner_h, w - corner_w:].reshape(-1, 3)
    sky_pixels = np.vstack([left_sky, right_sky])
    live_bg = np.median(sky_pixels, axis=0)

    if np.mean(live_bg) < 185:
        if SMOOTH_BG_COLOR is None:
            SMOOTH_BG_COLOR = live_bg.copy()
        else:
            SMOOTH_BG_COLOR = 0.90 * SMOOTH_BG_COLOR + 0.10 * live_bg

    bg_ref = SMOOTH_BG_COLOR if SMOOTH_BG_COLOR is not None else live_bg

    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]
    bg_b, bg_g, bg_r = bg_ref[0], bg_ref[1], bg_ref[2]

    red_vs_green = np.maximum(0.0, (r - g) - (bg_r - bg_g)) * 1.55
    blue_vs_green = np.maximum(0.0, (b - g) - (bg_b - bg_g)) * 1.35

    min_channel_gain = np.maximum(0.0, np.minimum.reduce([b - bg_b, g - bg_g, r - bg_r])) * 1.35
    rb_mean_gain = np.maximum(0.0, ((r - bg_r) + (b - bg_b)) * 0.5) * 1.25

    green_gain = np.maximum(0.0, g - bg_g)
    non_green_support = np.maximum(0.0, np.maximum(r - bg_r, b - bg_b))
    lime_green_block = np.where(
        (green_gain > 38.0) & (non_green_support > 14.0),
        green_gain * 1.15,
        np.maximum(0.0, green_gain - 32.0) * 1.10
    )

    signal = np.clip(
        np.maximum.reduce([red_vs_green, blue_vs_green, min_channel_gain, rb_mean_gain, lime_green_block]),
        0, 220
    ).astype(np.uint8)

    sky_noise = max(
        float(np.percentile(signal[:corner_h, :corner_w], 94)),
        float(np.percentile(signal[:corner_h, w - corner_w:], 94))
    )
    cutoff = float(np.clip(sky_noise + 14.0, 30.0, 68.0))
    return signal, cutoff


def scan_grid_with_strength(crop, cols, rows):
    signal, cutoff = compute_cell_signal_map(crop)

    h, w = crop.shape[:2]
    cell_w = w / float(cols)
    cell_h = h / float(rows)

    half_w = max(1, int(cell_w * 0.26))
    half_h = max(1, int(cell_h * 0.26))

    grid = np.zeros((rows, cols), dtype=int)
    cell_strength = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cx = int(c * cell_w + cell_w * 0.5)
            cy = int(r * cell_h + cell_h * 0.5)
            patch = signal[max(0, cy - half_h):min(h, cy + half_h + 1),
                           max(0, cx - half_w):min(w, cx + half_w + 1)]
            if patch.size > 0:
                val = float(np.percentile(patch, 70))
                cell_strength[r, c] = val
                if val >= cutoff:
                    grid[r, c] = 1

    return grid, cell_strength, cutoff


def clean_settled_board(settled_grid):
    cleaned = settled_grid.copy()
    rows, cols = cleaned.shape
    cleaned[:4, :] = 0

    supported = np.zeros_like(cleaned)
    supported[rows - 1, :] = cleaned[rows - 1, :]

    for r in range(rows - 2, 3, -1):
        row_cells = cleaned[r, :]
        if not np.any(row_cells):
            continue
        c = 0
        while c < cols:
            if row_cells[c] == 1:
                c_start = c
                while c < cols and row_cells[c] == 1:
                    c += 1
                c_end = c
                if np.any(supported[r + 1, c_start:c_end] == 1):
                    supported[r, c_start:c_end] = 1
            else:
                c += 1

    return supported


def classify_spawn_from_strength(grid, cell_strength, cutoff, search_rows=5):
    spawn_zone = cell_strength[:search_rows, 2:8].copy()
    peak_val = float(np.max(spawn_zone))
    if peak_val < cutoff * 0.90:
        return 'UNKNOWN', None

    adaptive_thr = max(cutoff * 0.82, peak_val * 0.52)
    spawn_bin = (spawn_zone >= adaptive_thr).astype(np.uint8)

    if int(np.sum(spawn_bin)) < 3:
        return 'UNKNOWN', None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spawn_bin, connectivity=8)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        top_row = stats[label, cv2.CC_STAT_TOP]
        if 3 <= area <= 8 and top_row <= 2:
            mean_str = float(np.mean(spawn_zone[labels == label]))
            if mean_str > best_score:
                best_score = mean_str
                best_label = label

    if best_label is None:
        return 'UNKNOWN', None

    comp_mask = (labels == best_label)
    rs, cs = np.where(comp_mask)
    min_r, max_r = int(rs.min()), int(rs.max())
    min_c, max_c = int(cs.min()), int(cs.max())

    sh = max_r - min_r + 1
    sw = max_c - min_c + 1

    if sw >= 4 and sh == 1:
        full_mask = np.zeros_like(grid)
        full_mask[min_r, 2 + min_c : 2 + min_c + 4] = 1
        return 'I', full_mask

    if sh < 2 or sw < 2:
        return 'UNKNOWN', None

    sub_str = spawn_zone[min_r:min_r + 2, min_c:max_c + 1]
    if sw >= 4:
        row_sums = np.sum(sub_str, axis=1)
        if max(row_sums) > 2.0 * min(row_sums):
            best_r = min_r + int(np.argmax(row_sums))
            full_mask = np.zeros_like(grid)
            full_mask[best_r, 2 + min_c : 2 + min_c + 4] = 1
            return 'I', full_mask

    if sw == 2:
        box_2x2 = spawn_zone[min_r:min_r + 2, min_c:min_c + 2]
        left_flank = float(np.max(spawn_zone[min_r:min_r + 2, min_c - 1])) if min_c > 0 else 0.0
        right_flank = float(np.max(spawn_zone[min_r:min_r + 2, max_c + 1])) if max_c < 5 else 0.0

        if float(np.min(box_2x2)) >= peak_val * 0.65 and left_flank < peak_val * 0.38 and right_flank < peak_val * 0.38:
            full_mask = np.zeros_like(grid)
            full_mask[min_r:min_r + 2, 2 + min_c : 2 + min_c + 2] = 1
            return 'O', full_mask
        if left_flank >= right_flank and min_c > 0:
            min_c -= 1
        elif max_c < 5:
            max_c += 1
        sw = max_c - min_c + 1

    best_3x2_name = 'UNKNOWN'
    best_3x2_score = -999.0
    best_r0, best_c0 = min_r, min_c

    c_start_lim = max(0, min(min_c, 3))
    c_end_lim = min(3, max_c - 2) if max_c >= 2 else c_start_lim

    for r0 in range(min_r, min(search_rows - 1, max_r + 1)):
        if r0 + 2 > search_rows:
            continue
        for c0 in range(c_start_lim, c_end_lim + 1):
            win_2x3 = spawn_zone[r0:r0 + 2, c0:c0 + 3]
            if win_2x3.shape != (2, 3):
                continue

            w_min = float(np.min(win_2x3))
            w_max = float(np.max(win_2x3))
            if w_max - w_min < 8.0:
                continue
            norm_win = (win_2x3 - w_min) / (w_max - w_min)

            for name in ('T', 'S', 'Z', 'J', 'L'):
                tmpl = PIECE_TEMPLATES[name]
                ones_mean = float(np.sum(norm_win * tmpl)) / 4.0
                zeros_mean = float(np.sum(norm_win * (1.0 - tmpl))) / 2.0
                score = ones_mean - zeros_mean
                if score > best_3x2_score:
                    best_3x2_score = score
                    best_3x2_name = name
                    best_r0, best_c0 = r0, c0

    if best_3x2_name != 'UNKNOWN' and best_3x2_score >= 0.16:
        full_mask = np.zeros_like(grid)
        full_mask[best_r0:best_r0 + 2, 2 + best_c0 : 2 + best_c0 + 3] = PIECE_TEMPLATES[best_3x2_name].astype(int)
        return best_3x2_name, full_mask

    return 'UNKNOWN', None


def detect_falling_shape(frame, rect):
    trimmed = get_trimmed_playfield_rect(rect)
    x, y, w, h = trimmed["left"], trimmed["top"], trimmed["width"], trimmed["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN', None, None

    grid, cell_strength, cutoff = scan_grid_with_strength(crop, BOARD_COLS, BOARD_ROWS)
    shape, mask_top = classify_spawn_from_strength(grid, cell_strength, cutoff, search_rows=5)

    if mask_top is None:
        return 'UNKNOWN', grid, np.zeros_like(grid)

    return shape, grid, mask_top


def detect_next_shape(frame, rect, block_size_px):
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    if w < 8 or h < 8 or block_size_px <= 0:
        return 'UNKNOWN', None

    pad_x = max(2, int(round(w * 0.10)))
    pad_y = max(2, int(round(h * 0.10)))
    crop = frame[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]
    if crop.size == 0:
        return 'UNKNOWN', None

    crop_f = crop.astype(np.float32)
    ch, cw = crop.shape[:2]

    bg_color = np.percentile(crop_f.reshape(-1, 3), 12, axis=0)

    diff_bgr = np.maximum(0.0, crop_f - bg_color)
    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]

    euclidean_dist = np.sqrt(np.sum(diff_bgr * diff_bgr, axis=2)) * 0.65
    red_boost = np.maximum(0.0, (r - g) - (bg_color[2] - bg_color[1])) * 1.40
    blue_boost = np.maximum(0.0, (b - g) - (bg_color[0] - bg_color[1])) * 1.25
    green_boost = np.maximum(0.0, g - bg_color[1]) * 0.90

    raw_energy = np.maximum.reduce([euclidean_dist, red_boost, blue_boost, green_boost])
    raw_energy[0, :] = 0
    raw_energy[-1, :] = 0
    raw_energy[:, 0] = 0
    raw_energy[:, -1] = 0

    e_max = float(np.percentile(raw_energy, 98))
    e_min = float(np.percentile(raw_energy, 15))
    if e_max - e_min < 25.0:
        return 'UNKNOWN', np.zeros((ch, cw), dtype=np.uint8)

    norm_energy = np.clip((raw_energy - e_min) / (e_max - e_min), 0.0, 1.0)
    signal_u8 = (norm_energy * 255.0).astype(np.uint8)

    _, bbox_thresh = cv2.threshold(signal_u8, 140, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bbox_thresh, connectivity=8)
    best_label = None
    best_area = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]
        if area >= 6 and (0.10 * cw <= cx <= 0.90 * cw) and (0.10 * ch <= cy <= 0.90 * ch):
            if area > best_area:
                best_area = area
                best_label = lbl

    if best_label is None:
        return 'UNKNOWN', bbox_thresh

    mask = (labels == best_label).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    tight_energy = norm_energy[min_y:max_y + 1, min_x:max_x + 1]
    th, tw = tight_energy.shape[:2]
    if th < 3 or tw < 4:
        return 'UNKNOWN', bbox_thresh

    aspect = tw / float(th)

    if aspect >= 2.10:
        clean_vis = np.zeros((2, 4), dtype=np.uint8)
        clean_vis[0, :] = 255
        return 'I', clean_vis

    subgrid_energy = np.zeros((2, 3), dtype=np.float32)
    for r_idx in range(2):
        for c_idx in range(3):
            y0 = int(round((r_idx + 0.18) * th / 2.0))
            y1 = max(y0 + 1, int(round((r_idx + 0.82) * th / 2.0)))
            x0 = int(round((c_idx + 0.18) * tw / 3.0))
            x1 = max(x0 + 1, int(round((c_idx + 0.82) * tw / 3.0)))
            patch = tight_energy[y0:y1, x0:x1]
            if patch.size > 0:
                subgrid_energy[r_idx, c_idx] = float(np.percentile(patch, 60))

    sorted_cells = np.sort(subgrid_energy.flatten())
    weakest_2_mean = float(np.mean(sorted_cells[:2]))
    strongest_4_mean = float(np.mean(sorted_cells[2:]))

    if weakest_2_mean >= 0.55 and (strongest_4_mean - weakest_2_mean) < 0.28 and aspect <= 1.42:
        clean_vis = np.ones((2, 2), dtype=np.uint8) * 255
        return 'O', clean_vis

    best_3x2_shape = 'UNKNOWN'
    best_3x2_score = -999.0
    for name in ('T', 'S', 'Z', 'J', 'L'):
        tmpl = PIECE_TEMPLATES[name]
        ones_mean = float(np.sum(subgrid_energy * tmpl)) / 4.0
        zeros_mean = float(np.sum(subgrid_energy * (1.0 - tmpl))) / 2.0
        diff_score = ones_mean - zeros_mean
        if diff_score > best_3x2_score:
            best_3x2_score = diff_score
            best_3x2_shape = name

    top4_vis = (PIECE_TEMPLATES[best_3x2_shape] * 255).astype(np.uint8) if best_3x2_shape in PIECE_TEMPLATES else bbox_thresh
    return best_3x2_shape, top4_vis


def main(args):
    calibration = load_calibration()
    playfield = calibration["playfield"]
    next_box = calibration["next_box"]
    source = args.source or calibration.get("source", "camera")

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

    win_name = "AI Champion Controller (Pi + OV5647)"
    if not args.headless:
        cv2.namedWindow(win_name)

    sequencer = ActionSequencer()
    current_instructions = []

    candidate_shape = 'UNKNOWN'
    streak_count = 0
    empty_frames = 0
    spawn_locked = False
    last_valid_next = 'T'
    next_vote_buffer = []
    best_grid_snapshot = None
    frame_counter = 0

    trimmed_pf = get_trimmed_playfield_rect(pf_rel)
    block_size_px = trimmed_pf["width"] / float(BOARD_COLS)

    print("\n🍓 AI Vision Engine Online (Raspberry Pi OV5647 + 4-Line Tetris Well Mode)! Waiting for pieces...\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue

        frame_counter += 1
        falling_shape, full_grid, falling_mask = detect_falling_shape(frame, pf_rel)
        next_shape, next_debug_mask = detect_next_shape(frame, nb_rel, block_size_px)

        if next_shape in PIECE_TEMPLATES:
            next_vote_buffer.append(next_shape)
            if len(next_vote_buffer) > 5:
                next_vote_buffer.pop(0)
            last_valid_next = max(set(next_vote_buffer), key=next_vote_buffer.count)

        raw_settled = np.where(falling_mask == 1, 0, full_grid) if full_grid is not None else None
        clean_settled = clean_settled_board(raw_settled) if raw_settled is not None else None

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
                if empty_frames >= 2:
                    spawn_locked = False
                    streak_count = 0
                    candidate_shape = 'UNKNOWN'

        if not spawn_locked and streak_count >= 2 and candidate_shape in PIECE_TEMPLATES:
            valid_next = last_valid_next

            needs_cleanup = np.any(best_grid_snapshot[:8, :] == 1)
            active_dna = CLEANUP_DNA if needs_cleanup else CHAMPION_DNA
            mode_label = "CLEANUP" if needs_cleanup else "TETRIS-WELL"

            t0 = time.perf_counter()
            best_move, _, _ = find_best_move(best_grid_snapshot.tolist(), candidate_shape, valid_next, active_dna)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            print(f"Verified Spawn: {candidate_shape} (Next: {valid_next}) [{mode_label} | {dt_ms:.1f} ms | Cache: {len(BOARD_CACHE)}]")

            if best_move:
                best_rot, best_col = best_move
                current_instructions = sequencer.get_full_sequence(candidate_shape, best_rot, best_col)
                print(f"AI COMMAND: {' -> '.join(current_instructions)}\n")

            spawn_locked = True

        # Over Raspberry Pi Connect screen share, only redraw the GUI every 2nd frame to save Pi 4 CPU!
        if not args.headless and (frame_counter % 2 == 0):
            eff_left = trimmed_pf["left"]
            eff_top = trimmed_pf["top"]
            eff_w = trimmed_pf["width"]
            eff_h = trimmed_pf["height"]

            cv2.rectangle(frame, (eff_left, eff_top), (eff_left + eff_w, eff_top + eff_h), (0, 255, 0), 2)
            cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                          (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)

            cv2.putText(frame, f"SPAWN: {candidate_shape}", (eff_left, max(20, eff_top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"NEXT: {last_valid_next}", (nb_rel["left"], max(20, nb_rel["top"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if full_grid is not None and clean_settled is not None:
                cell_w = eff_w / float(BOARD_COLS)
                cell_h = eff_h / float(BOARD_ROWS)
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        cx = int(eff_left + c * cell_w + cell_w * 0.5)
                        cy = int(eff_top + r * cell_h + cell_h * 0.5)
                        if falling_mask[r, c] == 1:
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        elif clean_settled[r, c] == 1:
                            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        else:
                            cv2.circle(frame, (cx, cy), 1, (100, 100, 100), -1)

            if next_debug_mask is not None and next_debug_mask.size > 0:
                dbg_vis = cv2.cvtColor(next_debug_mask, cv2.COLOR_GRAY2BGR)
                dbg_vis = cv2.resize(dbg_vis, (96, 64), interpolation=cv2.INTER_NEAREST)
                fh, fw = frame.shape[:2]
                if fh > 90 and fw > 120:
                    frame[fh - 74:fh - 10, 10:106] = dbg_vis
                    cv2.rectangle(frame, (10, fh - 74), (106, fh - 10), (0, 255, 255), 1)
                    cv2.putText(frame, f"NEXT MASK ({last_valid_next})", (10, fh - 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if current_instructions:
                instruction_text = "DO: " + " -> ".join(current_instructions)
                cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capturer.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "screen"], default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Disable GUI window for max FPS on Raspberry Pi")
    cli_args = parser.parse_args()
    main(cli_args)
