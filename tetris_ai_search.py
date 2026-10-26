from tetris_shapes import SHAPES, rotate_piece
from tetris_brain import evaluate_board
from tetris_simulator import get_drop_position, place_piece, clear_lines

# BEAM_WIDTH = 4 -> Depth-1.5 (Evaluates all Piece 1 moves, then tests Piece 2 ONLY on the Top 4!)
# Set BEAM_WIDTH = 0 if you ever want pure Depth-1 instant mode on the Raspberry Pi.
BEAM_WIDTH = 4
BOARD_CACHE = {}


def eval_board_cached(board, weights):
    """Memoizes the board penalty score (with 0 lines) so duplicate states aren't recalculated."""
    fingerprint = tuple(tuple(row) for row in board)
    if fingerprint in BOARD_CACHE:
        return BOARD_CACHE[fingerprint]

    if len(BOARD_CACHE) > 20000:
        BOARD_CACHE.clear()

    board_score, _, _, _, _, _ = evaluate_board(board, 0, weights)
    BOARD_CACHE[fingerprint] = board_score
    return board_score


def get_line_reward(lines_scored, weights):
    if lines_scored == 1:
        return weights['lines_1']
    elif lines_scored == 2:
        return weights['lines_2']
    elif lines_scored == 3:
        return weights['lines_3']
    elif lines_scored == 4:
        return weights['lines_4']
    return 0.0


def get_unique_rotations(piece_name):
    """Skips redundant rotations on symmetric pieces (O=1 state, I/S/Z=2 states, T/J/L=4 states)."""
    if piece_name == 'O':
        return (0,)
    elif piece_name in ('I', 'S', 'Z'):
        return (0, 1)
    return (0, 1, 2, 3)


def find_best_move(board, current_piece_name, next_piece_name, weights, beam_width=BEAM_WIDTH):
    if current_piece_name not in SHAPES:
        return None, -999999, None

    board_cols = len(board[0])
    current_shape = SHAPES[current_piece_name]
    valid_rots1 = get_unique_rotations(current_piece_name)

    stage1_candidates = []

    # --- LEVEL 1: Evaluate all unique moves for the CURRENT piece ---
    for rot1 in range(4):
        if rot1 in valid_rots1:
            piece_w1 = len(current_shape[0])
            for col1 in range(board_cols):
                if col1 + piece_w1 > board_cols:
                    continue

                landing_row1 = get_drop_position(board, current_shape, col1)
                if landing_row1 < 0:
                    continue

                placed_board1 = place_piece(board, current_shape, landing_row1, col1)
                simulated_board1, lines_scored1 = clear_lines(placed_board1)

                reward1 = get_line_reward(lines_scored1, weights)
                board_score1 = eval_board_cached(simulated_board1, weights)
                stage1_score = board_score1 + reward1

                stage1_candidates.append((
                    stage1_score,
                    reward1,
                    (rot1, col1),
                    placed_board1,
                    simulated_board1
                ))

        current_shape = rotate_piece(current_shape)

    if not stage1_candidates:
        return None, -999999, None

    # Sort Level 1 moves from best to worst
    stage1_candidates.sort(key=lambda x: x[0], reverse=True)

    # Fast fallback: if beam_width is 0 or next_piece_name is invalid, return the best Level 1 move
    if beam_width <= 0 or next_piece_name not in SHAPES:
        best_s1 = stage1_candidates[0]
        return best_s1[2], best_s1[0], best_s1[3]

    # --- LEVEL 1.5 (BEAM SEARCH): Test NEXT piece ONLY on the Top `beam_width` Level 1 moves! ---
    top_finalists = stage1_candidates[:beam_width]
    next_shape_original = SHAPES[next_piece_name]
    valid_rots2 = get_unique_rotations(next_piece_name)

    best_overall_score = -999999
    best_current_move = top_finalists[0][2]
    best_current_board_state = top_finalists[0][3]

    for stage1_score, reward1, move1, placed_board1, simulated_board1 in top_finalists:
        best_future_score = -999999
        next_shape = next_shape_original

        for rot2 in range(4):
            if rot2 in valid_rots2:
                piece_w2 = len(next_shape[0])
                for col2 in range(board_cols):
                    if col2 + piece_w2 > board_cols:
                        continue

                    landing_row2 = get_drop_position(simulated_board1, next_shape, col2)
                    if landing_row2 < 0:
                        continue

                    placed_board2 = place_piece(simulated_board1, next_shape, landing_row2, col2)
                    simulated_board2, lines_scored2 = clear_lines(placed_board2)

                    # --- THE LOOPHOLE FIX ---
                    reward2 = get_line_reward(lines_scored2, weights)
                    total_line_rewards = reward1 + reward2
                    board_score2 = eval_board_cached(simulated_board2, weights)
                    honest_future_score = board_score2 + total_line_rewards

                    if honest_future_score > best_future_score:
                        best_future_score = honest_future_score

            next_shape = rotate_piece(next_shape)

        # If all Piece 2 moves topped out, fall back to the Stage 1 score
        effective_score = best_future_score if best_future_score > -999999 else stage1_score

        if effective_score > best_overall_score:
            best_overall_score = effective_score
            best_current_move = move1
            best_current_board_state = placed_board1

    return best_current_move, best_overall_score, best_current_board_state