import random
from tetris_shapes import SHAPES
from tetris_simulator import clear_lines
from tetris_ai_search import find_best_move

def create_empty_board(rows=20, cols=10):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def play_headless_game(weights, piece_limit=500):
    board = create_empty_board()
    fitness_score = 0  # <--- Change 1: Start tracking the actual score directly
    pieces_placed = 0
    piece_names = list(SHAPES.keys())

    # Generate the initial queue
    current_piece = random.choice(piece_names)
    next_piece = random.choice(piece_names)

    while pieces_placed < piece_limit:
        move, move_score, best_board = find_best_move(board, current_piece, next_piece, weights)

        if move is None:
            break

        board, lines_cleared = clear_lines(best_board)
        
        # --- Change 2: EXPONENTIAL FITNESS SCORING ---
        if lines_cleared == 1:
            fitness_score += 100       # Small reward for surviving
        elif lines_cleared == 2:
            fitness_score += 300       # Okay reward
        elif lines_cleared == 3:
            fitness_score += 800       # Good reward
        elif lines_cleared == 4:
            fitness_score += 4000      # MASSIVE TETRIS JACKPOT!
            
        pieces_placed += 1

        # Shift the queue
        current_piece = next_piece
        next_piece = random.choice(piece_names)

        # Game over if they hit the ceiling
        if sum(board[0]) > 0:
            break
            
    # Add pieces placed at the end so surviving longer still breaks ties!
    final_score = fitness_score + pieces_placed
    return final_score