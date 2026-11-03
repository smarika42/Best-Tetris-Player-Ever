import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from tetris_ai_search import find_best_move, BOARD_CACHE, DEPTH_2_WIDTH, DEPTH_1_5_WIDTH

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

CHAMPION_DNA = {
    'lines_1': -200.0,
    'lines_2': -60.0,
    'lines_3': 100.0,
    'lines_4': 9000.0,
    'height': 60.0,
    'holes': 1250.0,
    'bumpiness': 200.0,
    'wells': 280.0
}

CLEANUP_DNA = {
    'lines_1': 700.0,
    'lines_2': 1500.0,
    'lines_3': 2800.0,
    'lines_4': 9000.0,
    'height': 100.0,
    'holes': 1100.0,
    'bumpiness': 170.0,
    'wells': 60.0
}

NES_SPAWN_LEFT_COL = {
    'I': {0: 3, 1: 5, 2: 3, 3: 5},
    'O': {0: 4, 1: 4, 2: 4, 3: 4},
    'T': {0: 4, 1: 4, 2: 4, 3: 5},
    'S': {0: 4, 1: 5, 2: 4, 3: 5},
    'Z': {0: 4, 1: 5, 2: 4, 3: 5},
    'J': {0: 4, 1: 4, 2: 4, 3: 5},
    'L': {0: 4, 1: 4, 2: 4, 3: 5}
}

try:
    import mss
except ImportError:
    mss = None

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

BOARD_COLS = 10
BOARD_ROWS = 20

# Arducam OV5647 fisheye barrel correction factor for grid sampling points.
# Press '[' or ']' in the live window to fine-tune curvature!
FISHEYE_K = -0.12

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
    Zero-Lag Threaded Camera Driver for Arducam OV5647:
    - Uses buffer_count=2 and a dedicated background reader thread so `read()` ALWAYS
      returns the most recent millisecond frame and never lags behind the TV monitor!
    """
    def __init__(self, index=0, width=640, height=480):
        self.use_picam = HAS_PICAMERA2
        self.picam2 = None
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        if self.use_picam:
            try:
                print("🍓 Initializing Zero-Lag Arducam OV5647 via Picamera2 (640x480)...")
                self.picam2 = Picamera2(camera_num=index)
                config = self.picam2.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"},
                    buffer_count=2,
                    controls={"FrameDurationLimits": (16666, 25000)}  # 40-60 FPS
                )
                self.picam2.configure(config)
                self.picam2.start()

                time.sleep(0.6)
                self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})
                print("🔒 OV5647 Exposure & White Balance locked + Zero-Lag Thread started!")
            except Exception as e:
                print(f"⚠️ Picamera2 init failed ({e}), falling back to cv2.VideoCapture({index})...")
                self.use_picam = False

        if not self.use_picam:
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Start continuous background frame drainer so old frames never pile up!
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            if self.use_picam and self.picam2 is not None:
                frame = self.picam2.capture_array()
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            elif self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=0.3)
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
    trim_x = max(1, int(round(w * 0.025)))
    trim_top = max(1, int(round(h * 0.015)))
    trim_bot = max(1, int(round(h * 0.015)))
    return {
        "left": x + trim_x,
        "top": y + trim_top,
        "width": max(10, w - 2 * trim_x),
        "height": max(20, h - trim_top - trim_bot)
    }


def get_fisheye_cell_center(c, r, cols, rows, w, h):
    """
    Maps ideal grid cell (c, r) to the curved fisheye pixel coordinate (cx, cy)
    on the Arducam OV5647 so top/bottom corner dots don't drift into the walls!
    """
    nx = ((c + 0.5) / float(cols) - 0.5) * 2.0  # -1..1
    ny = ((r + 0.5) / float(rows) - 0.5) * 2.0  # -1..1
    r2 = nx * nx + ny * ny
    scale = 1.0 + FISHEYE_K * r2
    cx = int(round((0.5 + 0.5 * nx * scale) * w))
    cy = int(round((0.5 + 0.5 * ny * scale) * h))
    return int(np.clip(cx, 1, w - 2)), int(np.clip(cy, 1, h - 2))


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

    red_vs_green = np.maximum(0.0, (r - g) - (bg_r - bg_g)) * 1.85
    red_abs_gain = np.maximum(0.0, (r - bg_r) - np.maximum(0.0, g - bg_g) * 0.45) * 1.65
    blue_vs_green = np.maximum(0.0, (b - g) - (bg_b - bg_g)) * 1.65
    blue_abs_gain = np.maximum(0.0, (b - bg_b) - np.maximum(0.0, g - bg_g) * 0.45) * 1.55

    min_channel_gain = np.maximum(0.0, np.minimum.reduce([b - bg_b, g - bg_g, r - bg_r])) * 1.50
    rb_mean_gain = np.maximum(0.0, ((r - bg_r) + (b - bg_b)) * 0.5) * 1.40

    green_gain = np.maximum(0.0, g - bg_g)
    non_green_support = np.maximum(0.0, np.maximum(r - bg_r, b - bg_b))
    lime_green_block = np.where(
        (green_gain > 30.0) & (non_green_support > 10.0),
        green_gain * 1.30,
        np.maximum(0.0, green_gain - 25.0) * 1.20
    )

    signal = np.clip(
        np.maximum.reduce([
            red_vs_green, red_abs_gain, blue_vs_green, blue_abs_gain,
            min_channel_gain, rb_mean_gain, lime_green_block
        ]),
        0, 220
    ).astype(np.uint8)

    sky_noise = max(
        float(np.percentile(signal[:corner_h, :corner_w], 94)),
        float(np.percentile(signal[:corner_h, w - corner_w:], 94))
    )
    cutoff = float(np.clip(sky_noise + 12.0, 26.0, 60.0))
    return signal, cutoff


def scan_grid_with_strength(crop, cols, rows):
    signal, cutoff = compute_cell_signal_map(crop)

    h, w = crop.shape[:2]
    cell_w = w / float(cols)
    cell_h = h / float(rows)

    half_w = max(1, int(cell_w * 0.25))
    half_h = max(1, int(cell_h * 0.25))

    grid = np.zeros((rows, cols), dtype=int)
    cell_strength = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        row_cutoff = cutoff if r < 8 else (cutoff * 0.78)
        for c in range(cols):
            cx, cy = get_fisheye_cell_center(c, r, cols, rows, w, h)
            patch = signal[max(0, cy - half_h):min(h, cy + half_h + 1),
                           max(0, cx - half_w):min(w, cx + half_w + 1)]
            if patch.size > 0:
                val = float(np.percentile(patch, 72))
                cell_strength[r, c] = val
                if val >= row_cutoff:
                    grid[r, c] = 1

    return grid, cell_strength, cutoff


def clean_settled_board(settled_grid):
    cleaned = settled_grid.copy()
    rows, cols = cleaned.shape
    cleaned[:4, :] = 0

    num_labels, labels = cv2.connectedComponents(cleaned.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        component_mask = (labels == label)
        touches_floor = np.any(component_mask[rows - 2:, :])
        if not touches_floor:
            cleaned[component_mask] = 0

    return cleaned


def classify_spawn_from_strength(grid, cell_strength, cutoff, expected_piece=None, search_rows=4):
    """
    Fast & Strict Top-of-Board Spawn Classifier (Rows 0..3):
    - Only looks in Rows 0..3 (`search_rows=4`) so once a piece drops past Row 3,
      the spawn zone immediately reads empty and resets `spawn_locked` for the next piece!
    """
    spawn_zone = cell_strength[:search_rows, 2:8].copy()
    peak_val = float(np.max(spawn_zone))
    if peak_val < cutoff * 0.85:
        return 'UNKNOWN', None

    adaptive_thr = max(cutoff * 0.78, peak_val * 0.48)
    spawn_bin = (spawn_zone >= adaptive_thr).astype(np.uint8)

    if int(np.sum(spawn_bin)) < 3:
        return 'UNKNOWN', None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spawn_bin, connectivity=8)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        top_row = stats[label, cv2.CC_STAT_TOP]
        if 3 <= area <= 9 and top_row <= 1:
            mean_str = float(np.mean(spawn_zone[labels == label]))
            if mean_str > best_score:
                best_score = mean_str
                best_label = label

    if best_label is None:
        return 'UNKNOWN', None

    comp_mask = (labels == best_label)
    rs, cs = np.where(comp_mask)
    min_r, max_r = int(rs.min()), int(rs.max())

    sh = max_r - min_r + 1

    # 1. Flat 4-wide 'I' bar in Columns 3, 4, 5, 6
    for r_idx in range(min_r, min(search_rows, max_r + 1)):
        row_4 = spawn_zone[r_idx, 1:5]
        other_rows_max = 0.0
        if r_idx + 1 < search_rows:
            other_rows_max = max(other_rows_max, float(np.max(spawn_zone[r_idx + 1, 1:5])))
        if r_idx - 1 >= 0:
            other_rows_max = max(other_rows_max, float(np.max(spawn_zone[r_idx - 1, 1:5])))

        if float(np.min(row_4)) >= cutoff * 0.72 and float(np.mean(row_4)) > other_rows_max * 1.55:
            full_mask = np.zeros_like(grid)
            full_mask[r_idx, 3:7] = 1
            return 'I', full_mask

    if sh < 2 or min_r + 2 > search_rows:
        return 'UNKNOWN', None

    # 2. 2x2 'O' piece in Columns 4 & 5
    col3_energy = float(np.max(spawn_zone[min_r:min_r + 2, 1]))
    col4_5_box = spawn_zone[min_r:min_r + 2, 2:4]
    col6_energy = float(np.max(spawn_zone[min_r:min_r + 2, 4]))

    if (
        col4_5_box.shape == (2, 2)
        and float(np.min(col4_5_box)) >= peak_val * 0.65
        and col3_energy < peak_val * 0.35
        and col6_energy < peak_val * 0.35
    ):
        full_mask = np.zeros_like(grid)
        full_mask[min_r:min_r + 2, 4:6] = 1
        return 'O', full_mask

    # 3. 3-wide pieces (T, S, Z, J, L) in Columns 4, 5, 6 (`c0 = 2`)
    best_3x2_name = 'UNKNOWN'
    best_3x2_score = -999.0
    best_r0, best_c0 = min_r, 2

    for r0 in range(max(0, min_r), min(search_rows - 1, max_r + 1)):
        for c0 in (2, 1):
            if r0 + 2 > search_rows or c0 + 3 > 6:
                continue
            win_2x3 = spawn_zone[r0:r0 + 2, c0:c0 + 3]
            w_min = float(np.min(win_2x3))
            w_max = float(np.max(win_2x3))
            if w_max - w_min < 8.0:
                continue
            norm_win = (win_2x3 - w_min) / (w_max - w_min)

            col_bias = 0.12 if c0 == 2 else 0.0

            for name in ('T', 'S', 'Z', 'J', 'L'):
                tmpl = PIECE_TEMPLATES[name]
                ones_mean = float(np.sum(norm_win * tmpl)) / 4.0
                zeros_mean = float(np.sum(norm_win * (1.0 - tmpl))) / 2.0
                prior_bonus = 0.18 if (expected_piece == name) else 0.0
                score = (ones_mean - zeros_mean) + col_bias + prior_bonus

                if score > best_3x2_score:
                    best_3x2_score = score
                    best_3x2_name = name
                    best_r0, best_c0 = r0, c0

    if best_3x2_name != 'UNKNOWN' and best_3x2_score >= 0.18:
        full_mask = np.zeros_like(grid)
        full_mask[best_r0:best_r0 + 2, 2 + best_c0 : 2 + best_c0 + 3] = PIECE_TEMPLATES[best_3x2_name].astype(int)
        return best_3x2_name, full_mask

    return 'UNKNOWN', None


def detect_falling_shape(frame, rect, expected_piece=None):
    trimmed = get_trimmed_playfield_rect(rect)
    x, y, w, h = trimmed["left"], trimmed["top"], trimmed["width"], trimmed["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN', None, None

    grid, cell_strength, cutoff = scan_grid_with_strength(crop, BOARD_COLS, BOARD_ROWS)
    shape, mask_top = classify_spawn_from_strength(
        grid, cell_strength, cutoff, expected_piece=expected_piece, search_rows=4
    )

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
    red_boost = np.maximum(0.0, (r - g) - (bg_color[2] - bg_color[1])) * 1.45
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
    global FISHEYE_K
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

    win_name = "AI Champion Controller (Zero-Lag + Fisheye Tuned)"
    if not args.headless:
        cv2.namedWindow(win_name)

    sequencer = ActionSequencer()
    current_instructions = []

    candidate_shape = 'UNKNOWN'
    streak_count = 0
    spawn_locked = False
    last_valid_next = 'T'
    expected_spawn_piece = None
    next_vote_buffer = []
    best_grid_snapshot = None
    frame_counter = 0

    trimmed_pf = get_trimmed_playfield_rect(pf_rel)
    block_size_px = trimmed_pf["width"] / float(BOARD_COLS)

    print("\n🍓 AI Vision Engine Online (Zero-Lag Threaded Capture + Fisheye Grid)! Waiting for pieces...")
    print("💡 Tip: Press '[' or ']' in the camera window to adjust fisheye curvature!\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.002)
            continue

        frame_counter += 1
        falling_shape, full_grid, falling_mask = detect_falling_shape(
            frame, pf_rel, expected_piece=expected_spawn_piece
        )
        next_shape, next_debug_mask = detect_next_shape(frame, nb_rel, block_size_px)

        if next_shape in PIECE_TEMPLATES:
            next_vote_buffer.append(next_shape)
            if len(next_vote_buffer) > 3:
                next_vote_buffer.pop(0)
            last_valid_next = max(set(next_vote_buffer), key=next_vote_buffer.count)

        raw_settled = np.where(falling_mask == 1, 0, full_grid) if full_grid is not None else None
        clean_settled = clean_settled_board(raw_settled) if raw_settled is not None else None

        if falling_shape != 'UNKNOWN':
            best_grid_snapshot = clean_settled.copy()
            if falling_shape == candidate_shape:
                streak_count += 1
            else:
                candidate_shape = falling_shape
                streak_count = 1
        else:
            # Instant unlock as soon as the piece leaves the top 3 spawn rows!
            if full_grid is not None and np.sum(full_grid[:3, 2:8]) == 0:
                if spawn_locked:
                    expected_spawn_piece = last_valid_next
                spawn_locked = False
                streak_count = 0
                candidate_shape = 'UNKNOWN'

        if not spawn_locked and streak_count >= 1 and candidate_shape in PIECE_TEMPLATES:
            # Lock on frame 1 if it matches expected_spawn_piece, or frame 2 otherwise!
            if streak_count >= 2 or candidate_shape == expected_spawn_piece:
                valid_next = last_valid_next

                top_half_occupied = bool(np.any(best_grid_snapshot[:10, :] == 1))
                active_beam_width = DEPTH_1_5_WIDTH if top_half_occupied else DEPTH_2_WIDTH
                depth_label = "DEPTH-1.5 (FAST)" if top_half_occupied else "DEPTH-2 (FULL)"

                needs_cleanup = bool(np.any(best_grid_snapshot[:8, :] == 1))
                active_dna = CLEANUP_DNA if needs_cleanup else CHAMPION_DNA
                mode_label = "CLEANUP" if needs_cleanup else "TETRIS-WELL"

                t0 = time.perf_counter()
                best_move, _, _ = find_best_move(
                    best_grid_snapshot.tolist(),
                    candidate_shape,
                    valid_next,
                    active_dna,
                    beam_width=active_beam_width
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0

                print(f"Verified Spawn: {candidate_shape} (Next: {valid_next}) [{depth_label} | {mode_label} | {dt_ms:.1f} ms | Cache: {len(BOARD_CACHE)}]")

                if best_move:
                    best_rot, best_col = best_move
                    current_instructions = sequencer.get_full_sequence(candidate_shape, best_rot, best_col)
                    print(f"AI COMMAND: {' -> '.join(current_instructions)}\n")

                spawn_locked = True

        # Redraw GUI every 3rd frame so Raspberry Pi Connect / VNC never lags the CPU!
        if not args.headless and (frame_counter % 3 == 0):
            eff_left = trimmed_pf["left"]
            eff_top = trimmed_pf["top"]
            eff_w = trimmed_pf["width"]
            eff_h = trimmed_pf["height"]

            cv2.rectangle(frame, (eff_left, eff_top), (eff_left + eff_w, eff_top + eff_h), (0, 255, 0), 2)
            mid_y = int(eff_top + eff_h * 0.5)
            cv2.line(frame, (eff_left, mid_y), (eff_left + eff_w, mid_y), (255, 200, 0), 1)

            cv2.rectangle(frame, (nb_rel["left"], nb_rel["top"]),
                          (nb_rel["left"] + nb_rel["width"], nb_rel["top"] + nb_rel["height"]), (0, 255, 255), 2)

            cv2.putText(frame, f"SPAWN: {candidate_shape}", (eff_left, max(20, eff_top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"NEXT: {last_valid_next}", (nb_rel["left"], max(20, nb_rel["top"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if full_grid is not None and clean_settled is not None:
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        gx, gy = get_fisheye_cell_center(c, r, BOARD_COLS, BOARD_ROWS, eff_w, eff_h)
                        cx, cy = eff_left + gx, eff_top + gy
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
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('['):
                FISHEYE_K -= 0.02
                print(f"🔍 Fisheye curvature adjusted: FISHEYE_K = {FISHEYE_K:.2f}")
            elif key == ord(']'):
                FISHEYE_K += 0.02
                print(f"🔍 Fisheye curvature adjusted: FISHEYE_K = {FISHEYE_K:.2f}")

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
