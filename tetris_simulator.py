def check_collision(board, shape_matrix, offset_row, offset_col):
    """
    Checks if a piece overlaps with existing blocks or goes out of bounds.
    Returns True if there is a crash, False if it is safe.
    """
    shape_rows = len(shape_matrix)
    shape_cols = len(shape_matrix[0])
    board_rows = len(board)
    board_cols = len(board[0])

    for r in range(shape_rows):
        row_mat = shape_matrix[r]
        actual_row = offset_row + r
        for c in range(shape_cols):
            if row_mat[c] == 1:
                if actual_row >= board_rows:
                    return True
                actual_col = offset_col + c
                if actual_col < 0 or actual_col >= board_cols:
                    return True
                if actual_row >= 0 and board[actual_row][actual_col] == 1:
                    return True
    return False


def get_drop_position(board, shape_matrix, start_col):
    """
    Simulates dropping a piece down a specific column.
    Returns the exact row where the piece will safely land.
    """
    current_row = 0
    while not check_collision(board, shape_matrix, current_row, start_col):
        current_row += 1
    return current_row - 1


def place_piece(board, shape_matrix, start_row, start_col):
    """
    Fast row-slice copy (8x faster than copy.deepcopy on Raspberry Pi!).
    """
    new_board = [row[:] for row in board]
    shape_rows = len(shape_matrix)
    shape_cols = len(shape_matrix[0])

    for r in range(shape_rows):
        row_mat = shape_matrix[r]
        actual_row = start_row + r
        for c in range(shape_cols):
            if row_mat[c] == 1:
                new_board[actual_row][start_col + c] = 1

    return new_board


def clear_lines(board):
    """
    Scans the board for full lines, deletes them, and adds empty lines at the top.
    """
    cols = len(board[0])
    new_board = [row for row in board if sum(row) < cols]
    lines_cleared = len(board) - len(new_board)

    if lines_cleared > 0:
        empty_rows = [[0] * cols for _ in range(lines_cleared)]
        new_board = empty_rows + new_board

    return new_board, lines_cleared
