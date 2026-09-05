# Webcam Tetris Placement Bot

Points a webcam at a monitor/TV running Tetris, reads the board, and
tells you the best rotation + column for each piece the instant it spawns.

## Setup

```bash
pip install opencv-python numpy
```

## 1. Calibrate the board corners (once, or whenever the camera/screen moves)

```bash
python calibrate_board.py
```

Click the 4 corners of the **playfield** (the 10-wide tall grid where
pieces fall — not the NEXT box, not the score) in this exact order:
top-left, top-right, bottom-right, bottom-left. A warped preview window
will pop up once you've clicked all 4 — it should look like a clean,
undistorted, top-down rectangle. If it looks skewed or cut off, press
`r` to reset and re-click more carefully. Press `c` first if the live
feed is too jittery to click accurately — it freezes the current frame.

Copy the printed `BOARD_CORNERS` list into `config.py`.

## 2. Verify board reading before trusting the full bot

```bash
python test_board_read.py
```

This shows the warped board live with a grid overlay — green cells are
what the code currently thinks are "filled". Watch it while pieces move
and lock. If cells flicker incorrectly or don't line up with the actual
blocks:
- Re-run calibration if the grid lines don't match the real block edges.
- Raise `FILL_THRESHOLD` in `config.py` if empty cells are being read as
  filled (e.g. from screen glare or a busy background image).
- Lower it if actual blocks aren't being detected as filled.

## 3. Run the bot

```bash
python main.py
```

Each time a new piece spawns, it prints one line of JSON to stdout, e.g.:

```json
{"piece": "T", "rotation": 2, "column": 4, "row": 17, "score": 3.21}
```

- `piece`: which tetromino spawned
- `rotation`: index into that piece's rotation list in `shapes.py` (0 =
  spawn orientation)
- `column`: the leftmost column its bounding box should end up in after
  rotating
- `row`: diagnostic — the row it would land on (0 = top)
- `score`: the heuristic score of that placement (higher = better)

Pipe it into whatever consumes it:
```bash
python main.py | python your_program.py
```

## How piece detection works

New pieces only ever appear right after the previous one locks, so the
bot watches a "spawn zone" near the top-middle of the board. The instant
a fresh tetromino-shaped cluster of cells appears there, it:
1. Identifies which piece it is by matching its cell pattern.
2. Treats the rest of the board (minus that piece) as the locked stack.
3. Runs the placement solver against that stack.

This avoids needing frame-perfect tracking of the piece as it falls.

## Known limitations / things to tune

- **Lighting and glare**: filming a screen with a webcam is much noisier
  than a direct screen capture. Dim room lighting and a matte (non-glare)
  monitor angle will help a lot.
- **Frame rate**: `main.py` polls roughly 30x/second; if your webcam or
  machine is slow, spawn events might be missed. Lower `SPAWN_STABLE_FRAMES`
  in `config.py` if pieces are dropping too fast to be caught reliably.
- **Rotation math is approximate**: `shapes.py` has standard tetromino
  rotation shapes, not exact SRS wall-kick behavior — fine for "what's the
  best final placement", not for simulating exact keystrokes to get there.
- **Solver is greedy** (one piece at a time, no lookahead to the piece
  after next). This is a reasonable starting point; if you want stronger
  play later, the natural upgrade is 2-piece lookahead using the same
  `solver.py` machinery.
