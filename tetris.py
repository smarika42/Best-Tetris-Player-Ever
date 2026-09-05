import cv2
import numpy as np
import time
from collections import deque, Counter

# ==========================================
# CONFIGURATION
# ==========================================
SPAWN_COLUMN = 3

TETROMINOES = {
    'I': [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'T': [
        np.array([[0, 1, 0], [1, 1, 1]]),
        np.array([[1, 0], [1, 1], [1, 0]]),
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1], [1, 1], [0, 1]])
    ],
    'S': [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])],
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


# ==========================================
# 1. PIECE DEBOUNCER
# ==========================================

class PieceDebouncer:
    def __init__(self, required_frames=2):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)
        self.confirmed_piece = 'UNKNOWN'

    def update(self, raw_detected_piece):
        self.history.append(raw_detected_piece)
        if len(self.history) == self.required_frames:
            first = self.history[0]
            if all(p == first for p in self.history):
                self.confirmed_piece = first
        return self.confirmed_piece


# ==========================================
# 2. DECISION STABILIZER
# ==========================================

class DecisionStabilizer:
    def __init__(self, buffer_size=2, consensus_threshold=2):
        self.buffer_size = buffer_size
        self.consensus_threshold = consensus_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.locked_decision = None
        self.last_piece_type = None

    def get_stable_action(self, piece_type, best_rot, move_offset, target_col):
        if piece_type != self.last_piece_type:
            self.buffer.clear()
            self.locked_decision = None
            self.last_piece_type = piece_type

        if self.locked_decision is not None:
            return self.locked_decision

        candidate = (best_rot, move_offset, target_col)
        self.buffer.append(candidate)

        if len(self.buffer) == self.buffer_size:
            most_common, count = Counter(self.buffer).most_common(1)[0]
            if count >= self.consensus_threshold:
                self.locked_decision = most_common
                return self.locked_decision

        return candidate


# ==========================================
# 3. INSTANT FULL-SEQUENCE GENERATOR
# ==========================================

class ActionSequencer:
    def get_full_sequence(self, piece_type, best_rot, move_offset):
        if piece_type == 'UNKNOWN':
            return ["WAITING FOR NEXT PIECE..."]

        actions = []
        for _ in range(best_rot):
            actions.append("ROTATE")

        if move_offset < 0:
            for _ in range(abs(move_offset)):
                actions.append("MOVE LEFT")
        elif move_offset > 0:
            for _ in range(move_offset):
                actions.append("MOVE RIGHT")

        actions.append("DROP")
        return actions


# ==========================================
# 4. AI DECISION ENGINE
# ==========================================

class TetrisAI:
    def __init__(self):
        self.w_height = -0.510066
        self.w_lines = +0.760666
        self.w_holes = -0.35663
        self.w_bumpiness = -0.184483

    def evaluate_board(self, board, lines_cleared):
        rows, cols = board.shape
        
        mask = board != 0
        has_blocks = mask.any(axis=0)
        column_heights = np.zeros(cols, dtype=int)
        column_heights[has_blocks] = rows - mask.argmax(axis=0)[has_blocks]

        aggregate_height = np.sum(column_heights)

        row_indices = np.arange(rows)[:, None]
        below_top = row_indices >= (rows - column_heights)[None, :]
        holes = np.sum(below_top & (board == 0))

        bumpiness = np.sum(np.abs(np.diff(column_heights)))

        return (self.w_height * aggregate_height +
                self.w_lines * lines_cleared +
                self.w_holes * holes +
                self.w_bumpiness * bumpiness)

    def simulate_drop(self, board, piece, col):
        p_rows, p_cols = piece.shape
        b_rows, b_cols = board.shape

        if col < 0 or (col + p_cols) > b_cols:
            return None, 0

        drop_row = 0
        while drop_row + p_rows <= b_rows:
            target_area = board[drop_row:drop_row + p_rows, col:col + p_cols]
            if np.any((piece == 1) & (target_area == 1)):
                break
            drop_row += 1
        
        drop_row -= 1
        if drop_row < 0:
            return None, 0

        temp_board = board.copy()
        temp_board[drop_row:drop_row + p_rows, col:col + p_cols] |= piece

        full_rows = np.all(temp_board == 1, axis=1)
        lines_cleared = np.sum(full_rows)

        if lines_cleared > 0:
            remaining_rows = temp_board[~full_rows]
            padding = np.zeros((lines_cleared, b_cols), dtype=int)
            temp_board = np.vstack((padding, remaining_rows))

        return temp_board, lines_cleared

    def get_best_action(self, board, piece_type):
        if piece_type not in TETROMINOES:
            return 0, 0, SPAWN_COLUMN

        best_score = -float('inf')
        best_rot = 0
        best_col = SPAWN_COLUMN

        orientations = TETROMINOES[piece_type]

        for rot_idx, piece in enumerate(orientations):
            _, p_cols = piece.shape
            max_valid_col = board.shape[1] - p_cols
            
            for col in range(0, max_valid_col + 1):
                sim_board, lines_cleared = self.simulate_drop(board, piece, col)
                
                if sim_board is not None:
                    score = self.evaluate_board(sim_board, lines_cleared)
                    if score > best_score:
                        best_score = score
                        best_rot = rot_idx
                        best_col = col

        move_offset = best_col - SPAWN_COLUMN
        return best_rot, move_offset, best_col


# ==========================================
# 5. MOBILE APP LAYOUT DETECTOR
# ==========================================

class BoardDetector:
    def __init__(self):
        self.screen_roi = (180, 50, 280, 380)
        self.board_rel = (0.18, 0.22, 0.64, 0.56)
        self.preview_rel = (0.84, 0.25, 0.14, 0.15)

        self.rows = 20
        self.cols = 10
        self.debug_counts = {}

    def _get_sub_rois(self):
        sx, sy, sw, sh = self.screen_roi

        brx, bry, brw, brh = self.board_rel
        bx = int(sx + brx * sw)
        by = int(sy + bry * sh)
        bw = int(brw * sw)
        bh = int(brh * sh)

        prx, pry, prw, prh = self.preview_rel
        px = int(sx + prx * sw)
        py = int(sy + pry * sh)
        pw = int(prw * sw)
        ph = int(prh * sh)

        return (bx, by, bw, bh), (px, py, pw, ph)

    def get_grid_state(self, frame, mask_top_rows=3):
        (bx, by, bw, bh), _ = self._get_sub_rois()
        board_crop = frame[by:by+bh, bx:bx+bw]
        
        if board_crop.size == 0:
            return np.zeros((self.rows, self.cols), dtype=int)

        hsv = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
        piece_mask = cv2.inRange(hsv, np.array([0, 50, 60]), np.array([180, 255, 255]))

        resized = cv2.resize(piece_mask, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        grid = (resized > 75).astype(int)

        grid[:mask_top_rows, :] = 0
        return grid

    def detect_piece_type_raw(self, frame):
        _, (px, py, pw, ph) = self._get_sub_rois()
        crop = frame[py:py+ph, px:px+pw]
        
        if crop.size == 0:
            return 'UNKNOWN'

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Refined non-overlapping HSV bounds tuned for phone displays
        color_ranges = {
            'I': [(np.array([80, 80, 100]), np.array([100, 255, 255]))],                 # Cyan
            'O': [(np.array([22, 80, 100]), np.array([35, 255, 255]))],                  # Yellow
            'T': [(np.array([130, 70, 100]), np.array([155, 255, 255]))],                # Purple
            'S': [(np.array([36, 70, 100]), np.array([75, 255, 255]))],                  # Green
            'Z': [                                                                       # Red (Dual Mask)
                (np.array([0, 80, 100]), np.array([10, 255, 255])),
                (np.array([170, 80, 100]), np.array([180, 255, 255]))
            ],
            'J': [(np.array([102, 80, 100]), np.array([125, 255, 255]))],                # Dark Blue
            'L': [(np.array([11, 80, 100]), np.array([21, 255, 255]))]                   # Orange
        }

        max_pixels = 0
        detected_shape = 'UNKNOWN'
        self.debug_counts = {}

        for shape, ranges in color_ranges.items():
            total_count = 0
            for (low, high) in ranges:
                mask = cv2.inRange(hsv, low, high)
                total_count += cv2.countNonZero(mask)
            
            self.debug_counts[shape] = total_count

            # Lower threshold (15 pixels) to catch smaller preview renders
            if total_count > max_pixels and total_count > 15:
                max_pixels = total_count
                detected_shape = shape

        return detected_shape

    def draw_overlays(self, frame, grid, actions, target_col, piece_type):
        sx, sy, sw, sh = self.screen_roi
        (bx, by, bw, bh), (px, py, pw, ph) = self._get_sub_rois()

        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
        cv2.putText(frame, "ALIGN PHONE SCREEN HERE", (sx, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        cv2.putText(frame, "NEXT", (px, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cell_h = bh / self.rows
        cell_w = bw / self.cols
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 1:
                    cx = int(bx + c * cell_w)
                    cy = int(by + r * cell_h)
                    cv2.rectangle(frame, (cx, cy), (int(cx + cell_w), int(cy + cell_h)), (0, 0, 255), -1)

        cv2.putText(frame, f"PIECE: {piece_type}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = 70
        for i, act in enumerate(actions, 1):
            text = f"{i}. {act}" if piece_type != 'UNKNOWN' else act
            color = (0, 255, 0) if act != "DROP" else (0, 165, 255)
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_offset += 25

        # On-screen Debug Stats for Piece Detection
        dbg_y = 35
        cv2.putText(frame, "COLOR MATCHES:", (frame.shape[1] - 180, dbg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        dbg_y += 20
        for shape, count in self.debug_counts.items():
            color = (0, 255, 0) if shape == piece_type else (180, 180, 180)
            cv2.putText(frame, f"{shape}: {count} px", (frame.shape[1] - 180, dbg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            dbg_y += 18

        tx = int(bx + target_col * cell_w + cell_w / 2)
        cv2.line(frame, (tx, by), (tx, by + bh), (255, 0, 255), 2)


# ==========================================
# 6. MAIN EXECUTION LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(0)
    
    detector = BoardDetector()
    debouncer = PieceDebouncer(required_frames=2)
    ai = TetrisAI()
    stabilizer = DecisionStabilizer(buffer_size=2, consensus_threshold=2)
    sequencer = ActionSequencer()

    last_printed_log = ""

    print("\n--- MOBILE TETRIS CV ENGINE STARTED ---")
    print("Align phone screen so whole game UI fits inside the CYAN box.")
    print("Press 'q' in video window to exit.\n")

    win_name = "Tetris CV AI Controller"
    cv2.namedWindow(win_name)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        raw_piece = detector.detect_piece_type_raw(frame)
        active_piece = debouncer.update(raw_piece)

        grid = detector.get_grid_state(frame)

        raw_rot, raw_move, raw_col = ai.get_best_action(grid, active_piece)

        best_rot, move_offset, target_col = stabilizer.get_stable_action(
            active_piece, raw_rot, raw_move, raw_col
        )

        actions = sequencer.get_full_sequence(active_piece, best_rot, move_offset)

        if active_piece != 'UNKNOWN':
            log = f"[{active_piece}] DO ALL: {' -> '.join(actions)}"
            if log != last_printed_log:
                print(log)
                last_printed_log = log

        detector.draw_overlays(frame, grid, actions, target_col, active_piece)
        
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()