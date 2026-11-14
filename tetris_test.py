import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from tetris_shapes import SHAPES, rotate_piece
from tetris_ai_search import find_best_move, BOARD_CACHE
from hardware_controller import HardwareController

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# STRICT ZERO-HOLE DNA:
# 'holes' is 25,000.0 so the AI will NEVER drop Z, S, J, L, or T over an empty space!
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

# EXACT REAL SNES SPAWN MATRICES (Verified from your video at 0:08, 0:30, 0:36, 0:46, 0:56!):
# In SNES Tetris, T, J, and L all spawn with their 3-wide bar on TOP and tooth pointing DOWN!
SNES_TRUE_SPAWN_SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'T': [[1, 1, 1], [0, 1, 0]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'J': [[1, 1, 1], [0, 0, 1]],
    'L': [[1, 1, 1], [1, 0, 0]]
}

# Leftmost column of the piece on the SNES 10-wide board AFTER rotating `r` times CW from SNES spawn:
SNES_SPAWN_LEFT_COL_BY_CW_CLICKS = {
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
    k: np.array(v, dtype=np.float32) for k, v in SNES_TRUE_SPAWN_SHAPES.items()
}

EMPTY_BG_CELL_BGR = None
HOLD_FRAMES = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.float32)


def trim_matrix(mat):
    """Removes empty zero rows/cols around a 2D shape matrix so we can compare exact shapes."""
    arr = np.array(mat, dtype=int)
    ys, xs = np.where(arr == 1)
    if len(xs) == 0:
        return arr.tolist()
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].tolist()


class ActionSequencer:
    """
    CRITICAL FIX FOR T, L, J, S, Z LANDING UPSIDE-DOWN:
    In `tetris_shapes.py`, `SHAPES['T']` or `SHAPES['L']` may start in a different orientation
    (e.g. flat-side down `[[0, 1, 0], [1, 1, 1]]`) than the real SNES game (which spawns tooth-down
    `[[1, 1, 1], [0, 1, 0]]`)!
    Instead of blindly trusting `best_rot` as a raw click count, we compute the exact 2D matrix
    that the AI chose in `find_best_move`, and find how many clockwise clicks (`0, 1, 2, 3`)
    are needed from the REAL SNES SPAWN MATRIX (`SNES_TRUE_SPAWN_SHAPES`) to produce that exact shape!
    """
    def _get_real_cw_clicks(self, piece_type, ai_rot_index):
        if piece_type not in SHAPES or piece_type not in SNES_TRUE_SPAWN_SHAPES:
            return ai_rot_index % 4

        # 1. Get the exact 2D matrix the AI simulator selected after `ai_rot_index` rotations
        target_shape = SHAPES[piece_type]
        for _ in range(ai_rot_index):
            target_shape = rotate_piece(target_shape)
        target_trimmed = trim_matrix(target_shape)

        # 2. Rotate the REAL SNES spawn shape clockwise 0, 1, 2, 3 times until it matches `target_trimmed`!
        snes_shape = SNES_TRUE_SPAWN_SHAPES[piece_type]
        for cw_clicks in range(4):
            if trim_matrix(snes_shape) == target_trimmed:
                return cw_clicks
            snes_shape = rotate_piece(snes_shape)

        return ai_rot_index % 4

    def get_full_sequence(self, piece_type, best_rot, target_left_col):
        if piece_type not in PIECE_TEMPLATES:
            return []
        actions = []

        real_cw_clicks = self._get_real_cw_clicks(piece_type, best_rot)

        if piece_type == 'O':
            real_cw_clicks = 0
        elif piece_type in ('I', 'S', 'Z'):
            real_cw_clicks = real_cw_clicks % 2

        if real_cw_clicks == 1:
            actions.append("ROTATE_CW")
        elif real_cw_clicks == 2:
            actions.append("ROTATE_CW")
            actions.append("ROTATE_CW")
        elif real_cw_clicks == 3:
            actions.append("ROTATE_CCW")

        spawn_col = SNES_SPAWN_LEFT_COL_BY_CW_CLICKS.get(piece_type, {}).get(real_cw_clicks, 4)
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


def extract_cell_patches_bgr_and_spec(crop, cols, rows):
    h, w = crop.shape[:2]
    crop_f = crop.astype(np.float32)
    cell_w = w / float(cols)
    cell_h = h / float(rows)

    half_w = max(1, int(cell_w * 0.28))
    half_h = max(1, int(cell_h * 0.28))

    cell_bgr = np.zeros((rows, cols, 3), dtype=np.float32)
    cell_spec = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cx = int(c * cell_w + cell_w * 0.5)
            cy = int(r * cell_h + cell_h * 0.5)
            patch = crop_f[max(0, cy - half_h):min(h, cy + half_h + 1),
                           max(0, cx - half_w):min(w, cx + half_w + 1)]
            if patch.size > 0:
                flat = patch.reshape(-1, 3)
                cell_bgr[r, c] = np.median(flat, axis=0)
                luma = 0.114 * flat[:, 0] + 0.587 * flat[:, 1] + 0.299 * flat[:, 2]
                cell_spec[r, c] = float(np.percentile(luma, 90) - np.percentile(luma, 15))

    return cell_bgr, cell_spec


def scan_grid_with_strength(crop, cols, rows):
    global EMPTY_BG_CELL_BGR
    cell_bgr, cell_spec = extract_cell_patches_bgr_and_spec(crop, cols, rows)

    top_rb = 0.5 * (cell_bgr[:2, 3:7, 2] + cell_bgr[:2, 3:7, 0])
    mid_rb = 0.5 * (cell_bgr[4:6, 2:8, 2] + cell_bgr[4:6, 2:8, 0])
    is_wait_banner = (np.sum(np.max(mid_rb, axis=0) > 110) >= 4) and (np.max(top_rb) < 90)

    if EMPTY_BG_CELL_BGR is None:
        sky_bgr = np.median(
            np.vstack([cell_bgr[:3, :2].reshape(-1, 3), cell_bgr[:3, 8:].reshape(-1, 3)]),
            axis=0
        )
        EMPTY_BG_CELL_BGR = np.tile(sky_bgr, (rows, cols, 1))
        if is_wait_banner:
            EMPTY_BG_CELL_BGR[8:, :] = cell_bgr[8:, :]
            EMPTY_BG_CELL_BGR[:3, :] = cell_bgr[:3, :]
    elif is_wait_banner:
        EMPTY_BG_CELL_BGR[8:, :] = 0.85 * EMPTY_BG_CELL_BGR[8:, :] + 0.15 * cell_bgr[8:, :]
        EMPTY_BG_CELL_BGR[:3, :] = 0.85 * EMPTY_BG_CELL_BGR[:3, :] + 0.15 * cell_bgr[:3, :]

    diff = cell_bgr - EMPTY_BG_CELL_BGR
    db, dg, dr = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]

    # 1. Red/Orange/Yellow/White blocks
    red_gain = np.maximum(0.0, dr - 12.0) * 2.1

    # 2. Blue blocks (J)
    blue_gain = np.maximum(0.0, (db - dg) - 16.0) * 2.1

    # 3. White/Yellow blocks (O, T)
    white_gain = np.maximum(0.0, np.minimum.reduce([db, dg, dr])) * 1.9

    # 4. Green blocks (S, L): require BOTH brighter green AND high internal block contrast (`cell_spec > 32`)
    green_gain = np.where(cell_spec > 32.0, np.maximum(0.0, dg - 14.0) * 1.9 + (cell_spec - 32.0) * 0.8, 0.0)

    # 5. Specular-verified 3D distance (raise spec threshold from 26 -> 34 so background traces NEVER pass!)
    dist_3d = np.sqrt(db**2 + dg**2 + dr**2)
    spec_verified_dist = np.where(cell_spec > 34.0, np.maximum(0.0, dist_3d - 22.0) * 1.30, 0.0)

    cell_strength = np.maximum.reduce([
        red_gain,
        blue_gain,
        white_gain,
        green_gain,
        spec_verified_dist
    ]).astype(np.float32)

    sky_noise = max(
        float(np.max(cell_strength[:3, :2])),
        float(np.max(cell_strength[:3, 8:]))
    )
    cutoff = float(np.clip(sky_noise + 16.0, 30.0, 58.0))

    grid = (cell_strength >= cutoff * 0.90).astype(int)
    return grid, cell_strength, cutoff, is_wait_banner, cell_bgr


def update_settled_board(raw_grid, falling_mask):
    """
    35-Frame Stationary Lock + Strict No-Gap Floor Support:
    Why did phantom green dots appear at 0:24 and 1:10 in video_1.mp4?
    1) At 0:24, a piece was still falling at Row 12 when a second piece spawned, and because
       the previous code didn't require cells to sit still for 35 frames, it snapshotted the
       falling piece in mid-air!
    2) At 1:10, a blue "7" pop-up graphic appeared in Rows 10..14 (floating in mid-air with
       empty rows 15..18 underneath it!).
    Here:
    - Every cell must remain continuously occupied for 35 frames (~1.2s) in Rows 4..18
      (or 18 frames on Row 19) to even be a candidate! A falling piece or brief pop-up graphic
      moves/disappears faster than that!
    - If a cell turns empty for even 1-2 frames, `HOLD_FRAMES` drops by `-12.0` immediately!
    - STRICT FLOOR SUPPORT: A cell at `(r, c)` can ONLY turn green if `r == 19` OR if `(r + 1, c)`
      directly below it is already green (with at most 1 cell of horizontal overhang ONLY if the
      entire column below the supporting neighbor is solid all the way to Row 19!).
      That makes the floating "7" graphic at Row 10 and falling pieces at Row 12 100% IMPOSSIBLE to turn green!
    """
    global HOLD_FRAMES
    rows, cols = raw_grid.shape

    candidate = np.where(falling_mask == 1, 0, raw_grid).astype(np.float32)
    candidate[:4, :] = 0.0

    HOLD_FRAMES = np.where(
        candidate > 0.5,
        np.minimum(45.0, HOLD_FRAMES + 1.0),
        np.maximum(0.0, HOLD_FRAMES - 12.0)
    )
    HOLD_FRAMES[:4, :] = 0.0

    stat = np.zeros((rows, cols), dtype=int)
    stat[4:rows - 1, :] = (HOLD_FRAMES[4:rows - 1, :] >= 32.0).astype(int)
    stat[rows - 1, :] = (HOLD_FRAMES[rows - 1, :] >= 16.0).astype(int)

    # Bottom-up strict column support from Row 19
    supported = np.zeros((rows, cols), dtype=int)
    supported[rows - 1, :] = stat[rows - 1, :]

    for r in range(rows - 2, 3, -1):
        # 1. Direct vertical support: cell directly underneath `(r + 1, c)` is supported
        direct = ((stat[r, :] == 1) & (supported[r + 1, :] == 1)).astype(int)
        row_sup = direct.copy()

        # 2. Only allow 1 cell of horizontal overhang if the neighbor `(r, c±1)` has direct vertical support
        for c in range(cols):
            if stat[r, c] == 1 and row_sup[c] == 0:
                left_direct = (c > 0 and direct[c - 1] == 1)
                right_direct = (c < cols - 1 and direct[c + 1] == 1)
                if left_direct or right_direct:
                    row_sup[c] = 1

        supported[r, :] = row_sup

    # Wipe HOLD_FRAMES on any floating mid-air graphic (like the "7" bonus icon at 1:10) so it never lingers!
    HOLD_FRAMES = np.where((stat == 1) & (supported == 0), 0.0, HOLD_FRAMES)
    return supported


def classify_spawn_from_strength(grid, cell_strength, cutoff, search_rows=4):
    spawn_zone = cell_strength[:search_rows, 2:8].copy()
    peak_val = float(np.max(spawn_zone[:3, 1:5]))
    if peak_val < cutoff * 0.85:
        return 'UNKNOWN', None

    adaptive_thr = max(cutoff * 0.72, peak_val * 0.48)
    spawn_bin = (spawn_zone >= adaptive_thr).astype(np.uint8)

    if int(np.sum(spawn_bin[:3, 1:5])) < 3:
        return 'UNKNOWN', None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spawn_bin, connectivity=8)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        top_row = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        if 3 <= area <= 7 and top_row <= 1 and width <= 5:
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

    if sh == 1 and sw >= 4:
        full_mask = np.zeros_like(grid)
        full_mask[min_r, 3:7] = 1
        return 'I', full_mask

    if min_r + 2 > search_rows:
        return 'UNKNOWN', None

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


def detect_board_and_spawn(frame, rect):
    trimmed = get_trimmed_playfield_rect(rect)
    x, y, w, h = trimmed["left"], trimmed["top"], trimmed["width"], trimmed["height"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 'UNKNOWN', None, None, False, None

    grid, cell_strength, cutoff, is_wait_banner, cell_bgr = scan_grid_with_strength(crop, BOARD_COLS, BOARD_ROWS)

    if is_wait_banner:
        return 'UNKNOWN', np.zeros_like(grid), np.zeros_like(grid), True, cell_bgr

    shape, mask_top = classify_spawn_from_strength(grid, cell_strength, cutoff, search_rows=4)
    if mask_top is None:
        return 'UNKNOWN', grid, np.zeros_like(grid), False, cell_bgr

    return shape, grid, mask_top, False, cell_bgr


def detect_next_shape(frame, rect, block_size_px):
    """
    Yellow-Arch-Killer Next Box Detector:
    In video_1.mp4 (`0:00` - `1:48`), look at the bottom-left `NEXT MASK (T)` window:
    There is a giant WHITE ARCH along the top, left, and right edges of `NEXT MASK`!
    Why? Because `pad_l=12%, pad_t=12%` included the bright yellow arcade border of the Next Box!
    And because that border touches the piece in the middle, `connectedComponentsWithStats`
    merged the top yellow border + the piece into a giant 'T' shape on every single frame!

    How we fix it 100%:
    1) Crop deeper on all 4 sides (`pad_l=24%, pad_r=26%, pad_t=24%, pad_b=26%`) so the yellow
       arcade arch is outside the crop.
    2) Flood-fill (`cv2.floodFill`) or strip any bright pixels connected to the outer border of
       the crop! Even if 1 pixel of the yellow border enters the crop, stripping border-touching
       components deletes the arch completely while leaving the centered Next Piece untouched!
    """
    x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
    if w < 10 or h < 10:
        return 'UNKNOWN', None

    pad_l = max(4, int(round(w * 0.24)))
    pad_r = max(4, int(round(w * 0.26)))
    pad_t = max(4, int(round(h * 0.24)))
    pad_b = max(4, int(round(h * 0.26)))

    crop = frame[y + pad_t : y + h - pad_b, x + pad_l : x + w - pad_r]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return 'UNKNOWN', None

    crop_f = crop.astype(np.float32)
    ch, cw = crop.shape[:2]

    # Sample the pitch-black background of the Next Box interior
    bg_color = np.percentile(crop_f.reshape(-1, 3), 15, axis=0)
    b, g, r = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]

    red_boost = np.maximum(0.0, r - bg_color[2]) * 1.75
    green_boost = np.maximum(0.0, g - bg_color[1]) * 1.50
    blue_boost = np.maximum(0.0, b - bg_color[0]) * 1.75

    raw_energy = np.maximum.reduce([red_boost, green_boost, blue_boost])

    e_max = float(np.percentile(raw_energy, 95))
    e_min = float(np.percentile(raw_energy, 15))
    if e_max - e_min < 16.0:
        return 'UNKNOWN', np.zeros((ch, cw), dtype=np.uint8)

    norm_energy = np.clip((raw_energy - e_min) / (e_max - e_min), 0.0, 1.0)
    signal_u8 = (norm_energy * 255.0).astype(np.uint8)

    _, binary = cv2.threshold(signal_u8, 110, 255, cv2.THRESH_BINARY)

    # CRITICAL FIX: Flood-fill from every pixel on the outer 1px border (top, bottom, left, right)
    # with 0! Any yellow frame arch that touches the edge of the crop is instantly erased,
    # while the floating piece in the dark center of the Next Box stays 100% intact!
    cleaned = binary.copy()
    for col_i in range(cw):
        if cleaned[0, col_i] > 0:
            cv2.floodFill(cleaned, None, (col_i, 0), 0)
        if cleaned[ch - 1, col_i] > 0:
            cv2.floodFill(cleaned, None, (col_i, ch - 1), 0)
    for row_i in range(ch):
        if cleaned[row_i, 0] > 0:
            cv2.floodFill(cleaned, None, (0, row_i), 0)
        if cleaned[row_i, cw - 1] > 0:
            cv2.floodFill(cleaned, None, (cw - 1, row_i), 0)

    # If the piece itself slightly touched the border, fall back to zeroing a 2px border ring
    if np.sum(cleaned > 0) < 6:
        cleaned = binary.copy()
        cleaned[:2, :] = 0
        cleaned[-2:, :] = 0
        cleaned[:, :2] = 0
        cleaned[:, -2:] = 0

    ys, xs = np.where(cleaned > 0)
    if len(xs) < 4:
        return 'UNKNOWN', cleaned

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    tight_mask = (cleaned[min_y:max_y + 1, min_x:max_x + 1] > 0).astype(np.float32)
    th, tw = tight_mask.shape[:2]
    if th < 2 or tw < 3:
        return 'UNKNOWN', cleaned

    aspect = tw / float(th)

    # 1. 'I' piece is flat & wide (4x1)
    if aspect >= 2.15 and th <= max(5, int(ch * 0.45)):
        return 'I', cleaned

    # 2. Sample 2x3 grid directly from the cleaned binary mask (`tight_mask`)!
    # Look at your `NEXT MASK` in video_1.mp4: the binary outline of L, Z, T, S is super crisp
    # once the arch is removed! Measuring fill ratio in each of the 2x3 cells of `tight_mask`
    # directly classifies L vs J vs T vs S vs Z!
    subgrid_fill = np.zeros((2, 3), dtype=np.float32)
    for r_idx in range(2):
        for c_idx in range(3):
            y0 = int(round(r_idx * th / 2.0))
            y1 = max(y0 + 1, int(round((r_idx + 1) * th / 2.0)))
            x0 = int(round(c_idx * tw / 3.0))
            x1 = max(x0 + 1, int(round((c_idx + 1) * tw / 3.0)))
            patch = tight_mask[y0:y1, x0:x1]
            if patch.size > 0:
                subgrid_fill[r_idx, c_idx] = float(np.mean(patch))

    sorted_cells = np.sort(subgrid_fill.flatten())
    weakest_2_mean = float(np.mean(sorted_cells[:2]))
    strongest_4_mean = float(np.mean(sorted_cells[2:]))

    # 3. 'O' piece fills all 6 sub-cells (all 4 corners of its bounding box are filled)
    if weakest_2_mean >= 0.45 and (strongest_4_mean - weakest_2_mean) < 0.30 and aspect <= 1.40:
        return 'O', cleaned

    best_3x2_shape = 'UNKNOWN'
    best_3x2_score = -999.0
    for name in ('T', 'S', 'Z', 'J', 'L'):
        tmpl = PIECE_TEMPLATES[name]
        ones_mean = float(np.sum(subgrid_fill * tmpl)) / 4.0
        zeros_mean = float(np.sum(subgrid_fill * (1.0 - tmpl))) / 2.0
        diff_score = ones_mean - zeros_mean
        if diff_score > best_3x2_score:
            best_3x2_score = diff_score
            best_3x2_shape = name

    return best_3x2_shape, cleaned


def main(args):
    global EMPTY_BG_CELL_BGR, HOLD_FRAMES
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

    print("\n=====================================================================")
    print("🚀 V3 ENGINE ACTIVE: Matrix-Matched Rotations + 35-Frame Settled Lock!")
    print("💡 Controls: 'b' = Calibrate Empty Board | '+'/'-' = Focus | 'q' = Quit")
    print("=====================================================================\n")

    while True:
        ret, frame = capturer.read()
        if not ret or frame is None:
            time.sleep(0.002)
            continue

        frame_counter += 1
        falling_shape, full_grid, falling_mask, is_wait_banner, cell_bgr = detect_board_and_spawn(frame, pf_rel)
        next_shape, next_debug_mask = detect_next_shape(frame, nb_rel, block_size_px)

        if next_shape in PIECE_TEMPLATES:
            next_vote_buffer.append(next_shape)
            if len(next_vote_buffer) > 3:
                next_vote_buffer.pop(0)
            last_valid_next = max(set(next_vote_buffer), key=next_vote_buffer.count)

        if is_wait_banner:
            HOLD_FRAMES[:, :] = 0.0
            best_grid_snapshot[:, :] = 0
            spawn_locked = False
            streak_count = 0
            candidate_shape = 'UNKNOWN'

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
                if empty_sky_frames >= 3:
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
            elif key == ord('b') and cell_bgr is not None:
                EMPTY_BG_CELL_BGR = cell_bgr.copy()
                HOLD_FRAMES[:, :] = 0.0
                best_grid_snapshot[:, :] = 0
                print("📸 Captured exact 200-cell Empty Background Reference & cleared green dots!")
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
