import random
import time
import os
import copy
import sys
from tetris_shapes import SHAPES, rotate_piece
from tetris_simulator import clear_lines, get_drop_position
from tetris_ai_search import find_best_move

# WINDOWS HACK: This tiny command enables ANSI escape codes in the Windows Terminal!
if os.name == 'nt':
    os.system('') 

def create_empty_board(rows=20, cols=10):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def print_frame(board, shape, current_row, current_col, action, lines):
    """Draws the board with the falling piece hovering in its current position."""
    temp_board = copy.deepcopy(board)
    
    # Stamp the falling piece onto our temporary drawing board using '2's
    for r in range(len(shape)):
        for c in range(len(shape[0])):
            if shape[r][c] == 1:
                actual_r = current_row + r
                actual_c = current_col + c
                if 0 <= actual_r < len(temp_board) and 0 <= actual_c < len(temp_board[0]):
                    temp_board[actual_r][actual_c] = 2 
                    
    # THE ANTI-FLICKER MAGIC: 
    # Instead of clearing the screen, we teleport the cursor to the top-left (0,0)
    # The new frame will instantly overwrite the old one without going blank!
    sys.stdout.write('\033[H')
    
    print(f"--- 🏆 CHAMPION AI TETRIS DEMO ---")
    print(f"Lines Cleared: {lines} | Current Action: {action}".ljust(45))
    
    for row in temp_board:
        # Solid blocks = [], Falling blocks = @@, Empty = .
        row_str = "".join(["[]" if cell == 1 else "@@" if cell == 2 else " ." for cell in row])
        print(row_str)
    print("-" * 31)

def animate_and_execute_move(board, piece_name, target_rotation, target_col, total_lines):
    """Translates the AI's decision into physical steps and animates them."""
    current_col = 3 # Standard Tetris spawn column
    current_row = 0
    current_shape = SHAPES[piece_name]
    
    # 1. GENERATE SOLENOID INSTRUCTIONS
    instructions = []
    for _ in range(target_rotation): instructions.append("ROTATE")
        
    col_diff = target_col - current_col
    if col_diff < 0:
        for _ in range(abs(col_diff)): instructions.append("LEFT")
    elif col_diff > 0:
        for _ in range(col_diff): instructions.append("RIGHT")
        
    instructions.append("HARD_DROP")

    # 2. ANIMATE THE HORIZONTAL MOVES & ROTATIONS
    for action in instructions:
        if action == "ROTATE":
            current_shape = rotate_piece(current_shape)
        elif action == "LEFT":
            current_col -= 1
        elif action == "RIGHT":
            current_col += 1
            
        if action != "HARD_DROP":
            print_frame(board, current_shape, current_row, current_col, action, total_lines)
            time.sleep(0.08) # Sped up slightly for a smoother visual!
            
    # 3. ANIMATE GRAVITY (The Drop)
    landing_row = get_drop_position(board, current_shape, target_col)
    while current_row < landing_row:
        current_row += 1
        print_frame(board, current_shape, current_row, target_col, "FALLING", total_lines)
        time.sleep(0.015) 

def play_game():
    board = create_empty_board()
    total_lines = 0
    piece_names = list(SHAPES.keys())
    
    # High-Roller DNA

    manual_weights = {
        'lines_1': -300.664,
        'lines_2': -91.714,
        'lines_3': 133.465,
        'lines_4': 1200.615,
        'height': 1.232,
        'holes': 6.876,
        'bumpiness': 2.864,
        'wells': 1.377,       # A nice little bonus that softens the height penalty, but doesn't override it
    }

    # Generate the first two pieces to start the queue
    current_piece = random.choice(piece_names)
    next_piece = random.choice(piece_names)

    while True:
        # Pass BOTH pieces to our new Depth-2 search!
        move, move_score, best_board = find_best_move(board, current_piece, next_piece, manual_weights)

        if move is None:
            print("\n🚨 GAME OVER! The AI topped out. 🚨")
            print(f"Final Score: {total_lines} lines cleared!")
            break

        target_rotation, target_col = move
        animate_and_execute_move(board, current_piece, target_rotation, target_col, total_lines)

        board, lines_cleared = clear_lines(best_board)
        total_lines += lines_cleared

        # Shift the queue! The next piece becomes the current piece, and we draw a new next piece.
        current_piece = next_piece
        next_piece = random.choice(piece_names)

        if sum(board[0]) > 0:
            print("\n🚨 GAME OVER! The stack reached the ceiling. 🚨")
            print(f"Final Score: {total_lines} lines cleared!")
            break

if __name__ == "__main__":
    play_game()

        #'lines_1': 100,
        #'lines_2': 300,
        #'lines_3': 500,
        #'lines_4': 1200,
        #'height': 0.51,
        #'holes': 3.56,
        #'bumpiness': 0.18 

        #'lines_1': 58.822,
        #'lines_2': 187.489,
        #'lines_3': 9.544,
        #'lines_4': 1011.115,
        #'height': 3.312,
        #'holes': 3.303,
        #'bumpiness': 1.104,
  