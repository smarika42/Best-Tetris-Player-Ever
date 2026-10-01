

def check_collision(board, shape_matrix, offset_row, offset_col):
    """
    Checks if a piece overlaps with existing blocks or goes out of bounds.
    Returns True if there is a crash, False if it is safe.
    """
    shape_rows = len(shape_matrix)
    shape_cols = len(shape_matrix[0])
    board_rows = len(board)
    board_cols = len(board[0])

    # Look at every single cell inside our tiny Tetris piece matrix
    for r in range(shape_rows):
        for c in range(shape_cols):
            
            # We only care about the solid blocks (1s) of the Tetris piece
            if shape_matrix[r][c] == 1:
                
                # Calculate exactly where this block is on the big main board
                actual_row = offset_row + r
                actual_col = offset_col + c

                # 1. Did it fall through the floor?
                if actual_row >= board_rows:
                    return True
                
                # 2. Did it hit the left or right wall?
                if actual_col < 0 or actual_col >= board_cols:
                    return True

                # 3. Did it hit an existing block on the board?
                # We check actual_row >= 0 so pieces can safely spawn slightly above the board
                if actual_row >= 0 and board[actual_row][actual_col] == 1:
                    return True

    # If the loop finishes and didn't hit a single thing, the space is clear!
    return False

def get_drop_position(board, shape_matrix, start_col):
    """
    Simulates dropping a piece down a specific column.
    Returns the exact row where the piece will safely land.
    """
    current_row = 0
    
    # Keep pushing the piece down as long as there is NO collision
    while not check_collision(board, shape_matrix, current_row, start_col):
        current_row += 1
        
    # The loop broke because we hit something! 
    # That means current_row is inside the floor/block. 
    # We step back up exactly 1 row to find the safe landing spot.
    safe_row = current_row - 1
    
    return safe_row


import copy

def place_piece(board, shape_matrix, start_row, start_col):
    """
    Creates a copy of the board and permanently stamps the piece into it.
    Returns the new board so the AI can score it!
    """
    # We create a deep copy so we don't accidentally ruin the original board!
    new_board = copy.deepcopy(board)
    
    shape_rows = len(shape_matrix)
    shape_cols = len(shape_matrix[0])
    
    for r in range(shape_rows):
        for c in range(shape_cols):
            # If the piece has a block here, stamp it onto the new board
            if shape_matrix[r][c] == 1:
                actual_row = start_row + r
                actual_col = start_col + c
                new_board[actual_row][actual_col] = 1
                
    return new_board

def clear_lines(board):
    """
    Scans the board for full lines, deletes them, and adds empty lines at the top.
    Returns the new updated board and the number of lines that were cleared!
    """
    new_board = []
    lines_cleared = 0
    cols = len(board[0])

    # 1. Check every row in the current board
    for row in board:
        # If the row is completely full of 1s...
        if sum(row) == cols:
            lines_cleared += 1  # We scored! Don't add this row to the new board.
        else:
            new_board.append(row) # Not full? Keep it and add it to the new board.

    # 2. Gravity! For every line we erased at the bottom, 
    # we need to inject a brand new empty row at the very top (Index 0).
    for _ in range(lines_cleared):
        empty_row = [0] * cols  # This creates a row of zeros like [0, 0, 0, 0, 0, 0]
        new_board.insert(0, empty_row)

    return new_board, lines_cleared

