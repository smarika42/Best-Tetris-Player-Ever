from tetris_shapes import SHAPES, rotate_piece
from tetris_brain import evaluate_board
from tetris_simulator import get_drop_position, place_piece, clear_lines 

def find_best_move(board, current_piece_name, next_piece_name, weights):
    best_overall_score = -999999
    best_current_move = None
    best_current_board_state = None

    current_shape = SHAPES[current_piece_name]
    next_shape_original = SHAPES[next_piece_name]

    # --- LEVEL 1: Test all moves for the CURRENT piece ---
    for rot1 in range(4):
        board_cols = len(board[0])
        
        for col1 in range(board_cols):
            if col1 + len(current_shape[0]) > board_cols:
                continue 

            landing_row1 = get_drop_position(board, current_shape, col1)
            if landing_row1 < 0:
                continue

            # This is the board after piece 1 lands
            placed_board1 = place_piece(board, current_shape, landing_row1, col1)
            simulated_board1, lines_scored1 = clear_lines(placed_board1)

            # --- LEVEL 2: Test all moves for the NEXT piece on this simulated board ---
            best_future_score = -999999
            next_shape = next_shape_original
            
            for rot2 in range(4):
                for col2 in range(board_cols):
                    if col2 + len(next_shape[0]) > board_cols:
                        continue
                        
                    landing_row2 = get_drop_position(simulated_board1, next_shape, col2)
                    if landing_row2 < 0:
                        continue
                        
                    placed_board2 = place_piece(simulated_board1, next_shape, landing_row2, col2)
                    simulated_board2, lines_scored2 = clear_lines(placed_board2)
                    
                    # --- THE LOOPHOLE FIX ---
                    # 1. Get the true reward for Piece 1
                    reward1 = 0
                    if lines_scored1 == 1: reward1 = weights['lines_1']
                    elif lines_scored1 == 2: reward1 = weights['lines_2']
                    elif lines_scored1 == 3: reward1 = weights['lines_3']
                    elif lines_scored1 == 4: reward1 = weights['lines_4']

                    # 2. Get the true reward for Piece 2
                    reward2 = 0
                    if lines_scored2 == 1: reward2 = weights['lines_1']
                    elif lines_scored2 == 2: reward2 = weights['lines_2']
                    elif lines_scored2 == 3: reward2 = weights['lines_3']
                    elif lines_scored2 == 4: reward2 = weights['lines_4']
                    
                    total_line_rewards = reward1 + reward2

                    # 3. Get the board penalties (pass 0 lines so we don't double-count rewards)
                    board_score, _, _, _, _, _ = evaluate_board(simulated_board2, 0, weights)
                    
                    # 4. Calculate the honest final score
                    honest_future_score = board_score + total_line_rewards
                    
                    if honest_future_score > best_future_score:
                        best_future_score = honest_future_score
                        
                next_shape = rotate_piece(next_shape)
            
            # If this Level 1 move sets up an amazing Level 2 future, lock it in!
            if best_future_score > best_overall_score:
                best_overall_score = best_future_score
                best_current_move = (rot1, col1)
                best_current_board_state = placed_board1

        current_shape = rotate_piece(current_shape)

    return best_current_move, best_overall_score, best_current_board_state