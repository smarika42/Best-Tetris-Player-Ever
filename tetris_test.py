import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from tetris_ai_search import find_best_move, BOARD_CACHE
from hardware_controller import HardwareController

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# STRICT ZERO-HOLE DNA:
# 'holes' is 25,000.0 so the AI will NEVER leave a hole under Z, S, J, L, or T!
CHAMPION_DNA = {
    'lines_1': -250.0,
    'lines_2': -80.0,
    'lines_3': 100.0,
    'lines_4': 12000.0,
    'height': 75.0,
    'holes': 25000.0,
    'bumpiness': 320.0,
    'wells': 300.0
}

CLEANUP_DNA = {
    'lines_1': 900.0,
    'lines_2': 2000.0,
    'lines_3': 3500.0,
    'lines_4': 12000.0,
    'height': 140.0,
    'holes': 20000.0,
    'bumpiness': 240.0,
    'wells': 60.0
}

# In 2-Player SNES Tetris, 3-wide pieces (T, S, Z, J, L) spawn in Columns 4, 5, 6
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

PIECE_TEMPLATES = {
    'I': np.array([[1, 1, 1, 1]], dtype=np.float32),
    'O': np.array([[1, 1], [1, 1]], dtype=np.float32),
    'T': np.array([[1, 1, 1], [0, 1, 0]], dtype=np.float32),
    'S': np.array([[0, 1, 1], [1, 1, 0]], dtype=np.float32),
    'Z': np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32),
    'J': {0: np.array([[1, 1, 1], [0, 0, 1]], dtype=np.float32)}[0],
    'L': {0: np.array([[1, 1, 1], [1, 0, 0]], dtype=np.float32)}[0]
}

SMOOTH_BG_COLOR = None
# Per-cell baseline background signal map (learns the fixed circuit-board traces so they NEVER become phantom dots!)
BG_CELL_BASELINE = None
HOLD_FRAMES = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.float32)


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
    def __init__(self, index=0, width=640, height=360, saved_lens_pos=None):
        self.use_picam = HAS_PICAMERA2
        self.picam2 = None
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.lens_pos = float(saved_lens_pos) if saved_lens_pos is not None else 2.2

        if self.use_picam:
            try:
                print("🍓 Initializing Wide-FOV Arducam IMX708 via Picamera2 (640x360 Full-Sensor)...")
                self.picam2 = Picamera2(camera_num=index)
                config = self.picam2.create_video_configuration(
                    sensor={"output_size": (2304, 1296)},
                    main={"size": (width, height), "format": "RGB888"},
                    buffer_count=2,
                    controls={"FrameDurationLimits": (16666, 25000)}
                )
                self.picam2.configure(config)
                self.picam2.start()

                if saved_lens_pos is not None:
                    try:
                        self.picam2.set_controls({"AfMode": 0, "LensPosition": self.lens_pos})
                        print(f"🎯 Restored saved manual focus (LensPosition = {self.lens_pos:.2f})")
                    except Exception:
                        pass
                else:
                    try:
                        self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                        print("🔍 Continuous Autofocus active during warmup...")
                    except Exception:
                        pass
                    time.sleep(1.2)

                self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})
                print("🔒 IMX708 Exposure & White Balance locked + Zero-Lag Thread started!")
            except Exception as e:
                print(f"⚠️ Picamera2 init failed ({e}), falling back to cv2.VideoCapture({index})...")
                self.use_picam = False

        if not self.use_picam:
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

    def adjust_focus(self, delta):
        if self.use_picam and self.picam2 is not None:
            self.lens_pos = max(0.0, min(10.0, self.lens_pos + delta))
            try:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": self.lens_pos})
                print(f"🎯 Manual Focus set: LensPosition = {self.lens_pos:.2f}")
            except Exception as e:
                print(f"⚠️ Focus adjustment error: {e}")

    def trigger_autofocus(self):
        if self.use_picam and self.picam2 is not None:
            try:
                self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                print("🔍 Switched to Continuous Autofocus!")
            except Exception:
                pass

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
    trim_x = max(2, int(round(w * 0.035)))
    trim_top = max(1, int(round(h * 0.012)))
    trim_bot = max(1, int(round(h * 0.010)))
    return {
        "left": x + trim_x,
        "top": y + trim_top,
        "width": max(10, w - 2 * trim_x),
        "height": max(20, h - trim_top - trim_bot)
    }


def compute_cell_signal_map(crop):
    """
    Specular-Highlight + Saturated-Block Signal Map:
    - Why did phantom dots appear at (Row 5, Col 9), (Row 9, Col 1), and (Row 14, Col 1) in video_1.mp4?
      Because those exact spots on the SNES background have dark-green/light-green circuit-board art!
    - What does EVERY real SNES Tetris block have that the green circuit-board art NEVER has?
      1) High Red or High Blue channel (Red I/Z, Orange T/L, Yellow O, White O, Blue J), OR
      2) A bright white specular highlight pixel inside the 8x8 tile (even Green S/L and Dark Blue J
         have bright white/cyan specular dots inside every single block!).
    """
    global SMOOTH_BG_COLOR
    h, w = crop.shape[:2]
    crop_f = crop.astype(np.float32)

    corner_h = max(4, int(h * 0.14))
    corner_w = max(4, int(w * 0.20))
    left_sky = crop_f[:corner_h, :corner_w].reshape(-1, 3)
    right_sky = crop_f[:corner_h, w - corner_w:].reshape(-1, 3)
    sky_pixels = np.vstack([left_sky, right_sky])
    live_bg = np.median(sky_pixels, axis=0)

    if np.mean(live_bg) < 190:
        if SMOOTH_BG_COLOR is None:
            SMOOTH_BG_COLOR = live_bg.copy()
        else:
            SMOOTH_BG_COLOR = 0.90 * SMOOTH_BG_COLOR + 0.10 * live_bg

    bg_ref = SMOOTH_BG_COLOR if SMOOTH_BG_COLOR is not None else live_bg

    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]
    bg_b, bg_g, bg_r = bg_ref[0], bg_ref[1], bg_ref[2]

    # 1. Red, Orange, Yellow, White pieces (I, Z, L, T, O): strong Red channel above background
    red_vs_green = np.maximum(0.0, (r - g) - (bg_r - bg_g)) * 2.10
    red_abs_gain = np.maximum(0.0, r - (bg_r + 12.0)) * 1.85

    # 2. Blue & Navy pieces (J): Blue channel stays high while Green drops sharply, so (b - g) >> (bg_b - bg_g)!
    # Note: Dark-green circuit background traces drop BOTH B and G together, so (b - g) does NOT trigger on background traces!
    blue_vs_green = np.maximum(0.0, (b - g) - (bg_b - bg_g) - 8.0) * 2.40
    blue_abs_gain = np.maximum(0.0, b - (bg_b + 12.0)) * 1.90

    # 3. White & Yellow pieces (O, T)
    min_channel_gain = np.maximum(0.0, np.minimum.reduce([b - bg_b, g - bg_g, r - bg_r])) * 1.80

    # 4. Green pieces (S, L): Real Green blocks on SNES have BOTH higher G AND bright white/yellow specular highlights
    # (where R and B also rise on the highlight dot), unlike background solder pads which only have G!
    green_with_highlight = np.maximum(0.0, (g - (bg_g + 22.0)) + np.maximum(0.0, r - bg_r) * 0.8) * 1.75

    signal = np.clip(
        np.maximum.reduce([
            red_vs_green, red_abs_gain,
            blue_vs_green, blue_abs_gain,
            min_channel_gain,
            green_with_highlight
        ]),
        0, 235
    ).astype(np.uint8)

    sky_noise = max(
        float(np.percentile(signal[:corner_h, :corner_w], 94)),
        float(np.percentile(signal[:corner_h, w - corner_w:], 94))
    )
    cutoff = float(np.clip(sky_noise + 14.0, 28.0, 56.0))
    return signal, cutoff


def scan_grid_with_strength(crop, cols, rows):
    global BG_CELL_BASELINE
    signal, cutoff = compute_cell_signal_map(crop)

    h, w = crop.shape[:2]
    cell_w = w / float(cols)
    cell_h = h / float(rows)

    half_w = max(1, int(cell_w * 0.28))
    half_h = max(1, int(cell_h * 0.28))

    grid = np.zeros((rows, cols), dtype=int)
    cell_strength = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cx = int(c * cell_w + cell_w * 0.5)
            cy = int(r * cell_h + cell_h * 0.5)
            patch = signal[max(0, cy - half_h):min(h, cy + half_h + 1),
                           max(0, cx - half_w):min(w, cx + half_w + 1)]
            if patch.size > 0:
                # 75th percentile catches both the solid body AND the bright specular highlight of every tile!
                cell_strength[r, c] = float(np.percentile(patch, 75))

    # Subtract per-cell static background baseline (eliminates fixed circuit-board pads at Row 5 Col 9, Row 9 Col 1, etc.!)
    if BG_CELL_BASELINE is not None:
        adj_strength = np.maximum(0.0, cell_strength - BG_CELL_BASELINE * 0.85)
    else:
        adj_strength = cell_strength

    for r in range(rows):
        row_cutoff = cutoff if r < 5 else (cutoff * 0.82)
        for c in range(cols):
            if adj_strength[r, c] >= row_cutoff:
                grid[r, c] = 1

    return grid, adj_strength, cutoff, cell_strength


def is_banner_present(raw_cell_strength, cutoff):
    banner_band = raw_cell_strength[3:7, 1:9]
    active_cols = int(np.sum(np.max(banner_band, axis=0) >= cutoff * 1.10))
    top_spawn_empty = float(np.max(raw_cell_strength[:2, 3:7])) < cutoff * 0.85
    return active_cols >= 6 and top_spawn_empty


def calibrate_empty_background(raw_cell_strength, is_wait_screen=False):
    """
    Automatically learns the static circuit-board background pattern (like Row 5 Col 9 and Row 9 Col 1)
    during the 'WAIT' screen at the start of the match (or when the user presses 'b'),
    so background art can NEVER trigger phantom green dots!
    """
    global BG_CELL_BASELINE
    clean_sample = raw_cell_strength.copy()
    if is_wait_screen:
        # Mask out Rows 3..7 where the word "WAIT" sits; Rows 8..19 are 100% empty background during WAIT!
        clean_sample[2:8, :] = 0.0
    if BG_CELL_BASELINE is None:
        BG_CELL_BASELINE = np.minimum(clean_sample, 48.0)
    else:
        BG_CELL_BASELINE = np.maximum(BG_CELL_BASELINE * 0.90, np.minimum(clean_sample, 48.0))


def update_settled_board(raw_grid, falling_mask):
    """
    Physics-Enforced Gravity Support Filter (Makes Mid-Air Phantom Clouds Impossible!):
    In video_1.mp4 at 0:46 - 1:40, phantom green dots floated in Rows 9..16 of Columns 0..2
    even though Rows 18..19 directly underneath them were EMPTY (grey)!
    Why? Because 8-connected components allowed a diagonal chain to hang off a single block in Col 3!

    Now we enforce REAL TETRIS GRAVITY from Row 19 upward:
    1) A cell must be stationary for >= 22 frames (or 14 frames on Row 19).
    2) Bottom-Up Support Pass (Row 19 up to Row 4):
       - Row 19 cells are supported by the floor.
       - A cell at `(r, c)` in Row `r < 19` is ONLY kept if:
         a) It sits directly on top of a settled block at `(r + 1, c)`, OR
         b) It is horizontally connected (within <= 2 columns on the same row `r`) to a cell
            that sits directly on top of a settled block at `(r + 1, c_neighbor)`!
       This allows real horizontal overhangs (like the top of a T, L, J, S, Z) while completely
       destroying floating diagonal phantom chains!
    """
    global HOLD_FRAMES
    rows, cols = raw_grid.shape

    candidate = np.where(falling_mask == 1, 0, raw_grid).astype(np.float32)
    candidate[:4, :] = 0.0

    HOLD_FRAMES = np.where(
        candidate > 0.5,
        np.minimum(35.0, HOLD_FRAMES + 1.0),
        np.maximum(0.0, HOLD_FRAMES - 10.0)
    )
    HOLD_FRAMES[:4, :] = 0.0

    stat = np.zeros((rows, cols), dtype=int)
    stat[4:rows - 1, :] = (HOLD_FRAMES[4:rows - 1, :] >= 22.0).astype(int)
    stat[rows - 1, :] = (HOLD_FRAMES[rows - 1, :] >= 14.0).astype(int)

    # Bottom-up physical gravity support pass
    supported = np.zeros((rows, cols), dtype=int)
    supported[rows - 1, :] = stat[rows - 1, :]

    for r in range(rows - 2, 3, -1):
        # Step 1: Cells in row `r` that sit directly on top of a supported cell in `r + 1`
        direct_support = (stat[r, :] == 1) & (supported[r + 1, :] == 1)
        row_sup = direct_support.astype(int)

        # Step 2: Allow up to 2 cells of horizontal overhang on the SAME row `r` (for T/L/J/S/Z/I roofs)
        # ONLY if they are part of a contiguous horizontal bar of occupied cells on row `r`
        for _ in range(2):
            for c in range(cols):
                if stat[r, c] == 1 and row_sup[c] == 0:
                    left_ok = (c > 0 and row_sup[c - 1] == 1)
                    right_ok = (c < cols - 1 and row_sup[c + 1] == 1)
                    if left_ok or right_ok:
                        row_sup[c] = 1

        supported[r, :] = row_sup

    # Clear HOLD_FRAMES for any unsupported floating phantom noise so it never accumulates
    HOLD_FRAMES = np.where((stat == 1) & (supported == 0), 0.0, HOLD_FRAMES)
    return supported


def classify_spawn_from_strength(grid, cell_strength, cutoff, search_rows=5):
    """
    100% Independent Geometry Spawn Classifier (Zero Next-Box Bias!):
    - Fixes the bug at 0:04 where 'L' (2 rows tall) was called 'I' (1 row tall)!
    - Fixes the bug at 0:30 where 'S' was called 'Z' due to Next-Box prior_bonus!
    """
    spawn_zone = cell_strength[:search_rows, 2:8].copy()  # Cols 2, 3, 4, 5, 6, 7 (width 6)
    peak_val = float(np.max(spawn_zone[:3, 1:5]))
    if peak_val < cutoff * 0.80:
        return 'UNKNOWN', None

    adaptive_thr = max(cutoff * 0.70, peak_val * 0.48)
    spawn_bin = (spawn_zone >= adaptive_thr).astype(np.uint8)

    if int(np.sum(spawn_bin[:4, 1:5])) < 3:
        return 'UNKNOWN', None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spawn_bin, connectivity=8)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        top_row = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        if 3 <= area <= 7 and top_row <= 2 and width <= 5:
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

    # 1. STRICT 'I' BAR CHECK:
    # An 'I' bar at spawn is ONLY 1 row tall (`sh == 1`) and 4 columns wide (`sw == 4` in Cols 3, 4, 5, 6)!
    # At 0:04 in video_1.mp4, 'L' had `sh == 2` (3 blocks in Row 0 + 1 block in Row 1), so `sh == 1` prevents 'L' from EVER being called 'I'!
    if sh == 1 and sw >= 4:
        full_mask = np.zeros_like(grid)
        full_mask[min_r, 3:7] = 1
        return 'I', full_mask

    # Also check if Row `min_r` has all 4 cells strong in Cols 3..6 AND the row below has 0 strong cells
    row0_4 = spawn_zone[min_r, 1:5]
    row1_4 = spawn_zone[min_r + 1, 1:5] if (min_r + 1 < search_rows) else np.zeros(4)
    if float(np.min(row0_4)) >= peak_val * 0.62 and float(np.max(row1_4)) < peak_val * 0.35:
        full_mask = np.zeros_like(grid)
        full_mask[min_r, 3:7] = 1
        return 'I', full_mask

    if min_r + 2 > search_rows:
        return 'UNKNOWN', None

    # 2. Check for 2x2 'O' piece in Columns 4 & 5 (indices 2 & 3 of `spawn_zone`)
    col3_energy = float(np.max(spawn_zone[min_r:min_r + 2, 1]))
    col4_5_box = spawn_zone[min_r:min_r + 2, 2:4]
    col6_energy = float(np.max(spawn_zone[min_r:min_r + 2, 4]))

    if (
        col4_5_box.shape == (2, 2)
        and float(np.min(col4_5_box)) >= peak_val * 0.58
        and col3_energy < peak_val * 0.40
        and col6_energy < peak_val * 0.40
    ):
        full_mask = np.zeros_like(grid)
        full_mask[min_r:min_r + 2, 4:6] = 1
        return 'O', full_mask

    # 3. 3-wide pieces (T, S, Z, J, L) in Columns 4, 5, 6 (`c0 = 2` in `spawn_zone`)
    # Zero prior_bonus so Next Box never corrupts the Playfield Spawn classification!
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
            if w_max - w_min < 6.0:
                continue
            norm_win = (win_2x3 - w_min) / (w_max - w_min)

            col_bias = 0.16 if c0 == 2 else 0.0

            for name in ('T', 'S', 'Z', 'J', 'L'):
                tmpl = PIECE_TEMPLATES[name]
                ones_mean = float(np.sum(norm_win * tmpl)) / 4.0
                zeros_mean = float(np.sum(norm_win * (1.0 - tmpl))) / 2.0
                score = (ones_mean - zeros_mean) + col_bias

                if score > best_3x2_score:
                    best_3x2_score = score
                    best_3x2_name = name
                    best_r0, best_c0 = r0, c0

    if best_3x2_name != 'UNKNOWN' and best_3x2_score >= 0.18:
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

    grid, adj_strength, cutoff, raw_cell_strength = scan_grid_with_strength(crop, BOARD_COLS, BOARD_ROWS)

    if is_banner_present(raw_cell_strength, cutoff):
        # While "WAIT" is on screen at 0:00, automatically learn the empty lower-board circuit traces!
        calibrate_empty_background(raw_cell_strength, is_wait_screen=True)
        return 'UNKNOWN', np.zeros_like(grid), np.zeros_like(grid)

    shape, mask_top = classify_spawn_from_strength(grid, adj_strength, cutoff, search_rows=5)

    if mask_top is None:
        return 'UNKNOWN', grid, np.zeros_like(grid)

    return shape, grid, mask_top


def detect_next_shape(frame, rect, block_size_px):
    """
    Auto-Interior Next Box Detector (Fixes the bottom-right border corner that caused 'NEXT MASK (Z)' and 'NEXT MASK (L)'!):
    - Look at `NEXT MASK (Z)` at 0:00 - 0:34 in video_1.mp4: the bottom-right corner of the white/yellow
      arcade frame entered the Next Box crop!
    - Here, we automatically strip out any outer border rows/columns whose median brightness exceeds the
      dark-teal Next Box background, locking strictly onto the dark interior window before finding the piece!
    """
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    if w < 10 or h < 10 or block_size_px <= 0:
        return 'UNKNOWN', None

    # Initial 15% trim
    pad_x = max(2, int(round(w * 0.15)))
    pad_y = max(2, int(round(h * 0.15)))
    rough = frame[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]
    if rough.size == 0 or rough.shape[0] < 8 or rough.shape[1] < 8:
        return 'UNKNOWN', None

    # Auto-trim any bright yellow/white border rows/columns on all 4 sides!
    # The interior background of the Next Box is very dark (gray < 75), whereas the yellow/white border is bright (> 130).
    rough_gray = cv2.cvtColor(rough, cv2.COLOR_BGR2GRAY)
    rh, rw = rough_gray.shape

    r_start, r_end = 0, rh
    c_start, c_end = 0, rw

    # Strip top/bottom border rows where >40% of the row is bright frame
    while r_start < rh // 3 and np.percentile(rough_gray[r_start, :], 65) > 115:
        r_start += 1
    while r_end > (2 * rh) // 3 and np.percentile(rough_gray[r_end - 1, :], 65) > 115:
        r_end -= 1
    # Strip left/right border cols where >40% of the col is bright frame
    while c_start < rw // 3 and np.percentile(rough_gray[:, c_start], 65) > 115:
        c_start += 1
    while c_end > (2 * rw) // 3 and np.percentile(rough_gray[:, c_end - 1], 65) > 115:
        c_end -= 1

    # Add 2 extra pixels of margin inside the detected dark window so bevel glow is 100% gone
    r_start = min(r_start + 2, rh - 4)
    r_end = max(r_start + 4, r_end - 2)
    c_start = min(c_start + 2, rw - 4)
    c_end = max(c_start + 4, c_end - 2)

    crop = rough[r_start:r_end, c_start:c_end]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 6:
        return 'UNKNOWN', None

    crop_f = crop.astype(np.float32)
    ch, cw = crop.shape[:2]

    # The interior of the Next Box is dark teal/black (percentile 20 is the exact background)
    bg_color = np.percentile(crop_f.reshape(-1, 3), 20, axis=0)
    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]

    # Any block inside the dark Next Box is MUCH brighter in R, G, or B than the dark interior!
    red_boost = np.maximum(0.0, r - bg_color[2]) * 1.75
    green_boost = np.maximum(0.0, g - bg_color[1]) * 1.50
    blue_boost = np.maximum(0.0, b - bg_color[0]) * 1.75
    luma_boost = np.maximum(0.0, (r + g + b) - np.sum(bg_color)) * 0.65

    raw_energy = np.maximum.reduce([red_boost, green_boost, blue_boost, luma_boost])
    raw_energy[0, :] = 0
    raw_energy[-1, :] = 0
    raw_energy[:, 0] = 0
    raw_energy[:, -1] = 0

    e_max = float(np.percentile(raw_energy, 96))
    e_min = float(np.percentile(raw_energy, 15))
    if e_max - e_min < 18.0:
        return 'UNKNOWN', np.zeros((ch, cw), dtype=np.uint8)

    norm_energy = np.clip((raw_energy - e_min) / (e_max - e_min), 0.0, 1.0)
    signal_u8 = (norm_energy * 255.0).astype(np.uint8)

    _, bbox_thresh = cv2.threshold(signal_u8, 105, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bbox_thresh, connectivity=8)
    best_label = None
    best_area = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]
        if area >= 5 and (0.12 * cw <= cx <= 0.88 * cw) and (0.12 * ch <= cy <= 0.88 * ch):
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
    if th < 2 or tw < 3:
        return 'UNKNOWN', bbox_thresh

    aspect = tw / float(th)

    # 1. 'I' piece is flat & wide (4x1)
    if aspect >= 2.10 and th <= max(5, int(ch * 0.45)):
        clean_vis = np.zeros((2, 4), dtype=np.uint8)
        clean_vis[0, :] = 255
        return 'I', clean_vis

    # 2. Sample 2x3 grid of the tight bounding box
    subgrid_energy = np.zeros((2, 3), dtype=np.float32)
    for r_idx in range(2):
        for c_idx in range(3):
            y0 = int(round((r_idx + 0.12) * th / 2.0))
            y1 = max(y0 + 1, int(round((r_idx + 0.88) * th / 2.0)))
            x0 = int(round((c_idx + 0.12) * tw / 3.0))
            x1 = max(x0 + 1, int(round((c_idx + 0.88) * tw / 3.0)))
            patch = tight_energy[y0:y1, x0:x1]
            if patch.size > 0:
                subgrid_energy[r_idx, c_idx] = float(np.percentile(patch, 68))

    sorted_cells = np.sort(subgrid_energy.flatten())
    weakest_2_mean = float(np.mean(sorted_cells[:2]))
    strongest_4_mean = float(np.mean(sorted_cells[2:]))

    # 3. 'O' piece fills all 4 corners of its 2x2 bounding box
    if weakest_2_mean >= 0.52 and (strongest_4_mean - weakest_2_mean) < 0.28 and aspect <= 1.38:
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
        capturer = CameraCapture(camera_index, saved_lens_pos=calibration.get("lens_position"))
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

    win_name = "AI Champion Controller (Pi + IMX708)"
    if not args.headless:
        cv2.namedWindow(win_name)

    sequencer = ActionSequencer()
    hw_controller = HardwareController(use_solenoid_timings=args.solenoids)
    current_instructions = []

    candidate_shape = 'UNKNOWN'
    streak_count = 0
    spawn_locked = False
    empty_sky_frames = 0
    last_valid_next = 'T'
    next_vote_buffer = []
    best_grid_snapshot = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    frame_counter = 0

    trimmed_pf = get_trimmed_playfield_rect(pf_rel)
    block_size_px = trimmed_pf["width"] / float(BOARD_COLS)

    print("\n🍓 AI Vision Engine Online (Physics Gravity Support + Auto-Interior Next Box + Pure Spawn)!")
    print("💡 Controls: '+' / '-' = Manual Focus | 'a' = Autofocus | 'b' = Clear Phantom Dots | 't' = Test LEDs | 'q' = Quit\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.002)
            continue

        frame_counter += 1
        falling_shape, full_grid, falling_mask = detect_falling_shape(frame, pf_rel)
        next_shape, next_debug_mask = detect_next_shape(frame, nb_rel, block_size_px)

        if next_shape in PIECE_TEMPLATES:
            next_vote_buffer.append(next_shape)
            if len(next_vote_buffer) > 3:
                next_vote_buffer.pop(0)
            last_valid_next = max(set(next_vote_buffer), key=next_vote_buffer.count)

        clean_settled = (
            update_settled_board(full_grid, falling_mask)
            if full_grid is not None else None
        )

        if falling_shape != 'UNKNOWN':
            empty_sky_frames = 0
            if not spawn_locked and clean_settled is not None:
                best_grid_snapshot = clean_settled.copy()
            if falling_shape == candidate_shape:
                streak_count += 1
            else:
                candidate_shape = falling_shape
                streak_count = 1
        else:
            if full_grid is not None and int(np.sum(full_grid[:2, 3:7])) == 0:
                empty_sky_frames += 1
                if empty_sky_frames >= 2:
                    if clean_settled is not None:
                        best_grid_snapshot = clean_settled.copy()
                    spawn_locked = False
                    streak_count = 0
                    candidate_shape = 'UNKNOWN'

        if not spawn_locked and streak_count >= 2 and candidate_shape in PIECE_TEMPLATES:
            valid_next = last_valid_next

            top_half_occupied = bool(np.any(best_grid_snapshot[:10, :] == 1))
            active_beam_width = 4 if top_half_occupied else 12
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
                hw_controller.dispatch_sequence(current_instructions)

            spawn_locked = True

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
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                HOLD_FRAMES[:, :] = 0.0
                print("🧹 Cleared all settled stack memory & reset phantom dots!")
            elif key == ord('t'):
                demo_seq = ["ROTATE_CW", "LEFT", "LEFT", "DROP"]
                print(f"🧪 Manual LED Test Triggered: {' -> '.join(demo_seq)}")
                hw_controller.dispatch_sequence(demo_seq)
            elif key in (ord('+'), ord('=')) and isinstance(capturer, CameraCapture):
                capturer.adjust_focus(0.25)
            elif key == ord('-') and isinstance(capturer, CameraCapture):
                capturer.adjust_focus(-0.25)
            elif key == ord('a') and isinstance(capturer, CameraCapture):
                capturer.trigger_autofocus()

    capturer.release()
    hw_controller.cleanup()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "screen"], default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Disable GUI window for max FPS on Raspberry Pi")
    parser.add_argument("--solenoids", action="store_true", help="Use fast 35ms solenoid timings instead of 120ms LED timings")
    cli_args = parser.parse_args()
    main(cli_args)
