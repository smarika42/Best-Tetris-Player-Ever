def calculate_board_height(board):
    rows = len(board)
    cols = len(board[0])
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if board[r][c] == 1:
                heights[c] = rows - r
                break
    return sum(heights) + (max(heights) * 2)


def count_holes(board):
    total_holes = 0
    rows = len(board)
    cols = len(board[0])

    for col in range(cols):
        block_found = False
        for row in range(rows):
            if board[row][col] == 1:
                block_found = True
            elif board[row][col] == 0 and block_found:
                total_holes += 1
    return total_holes


def calculate_bumpiness(board):
    rows = len(board)
    cols = len(board[0])
    col_heights = [0] * cols

    for col in range(cols):
        for row in range(rows):
            if board[row][col] == 1:
                col_heights[col] = rows - row
                break

    # Measure bumpiness across the main stack (Columns 0 to 8) so it stays flat like a table
    total_bumpiness = 0
    for i in range(cols - 2):
        diff = abs(col_heights[i] - col_heights[i + 1])
        total_bumpiness += diff

    # Gentle penalty if Column 9 becomes taller than Column 8
    if col_heights[cols - 1] > col_heights[cols - 2]:
        total_bumpiness += (col_heights[cols - 1] - col_heights[cols - 2]) * 2

    return total_bumpiness


def calculate_wells(board):
    rows = len(board)
    cols = len(board[0])
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if board[r][c] == 1:
                heights[c] = rows - r
                break

    # Reward keeping Column 9 lower than the main stack (Right-Side Tetris Well)
    min_main_stack = min(heights[:cols - 1])
    right_well_depth = max(0, min_main_stack - heights[cols - 1])
    return min(right_well_depth, 4)


def evaluate_board(board, lines_cleared, weights):
    height = calculate_board_height(board)
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(board)
    wells = calculate_wells(board)

    line_reward = 0
    if lines_cleared == 1: line_reward = weights['lines_1']
    elif lines_cleared == 2: line_reward = weights['lines_2']
    elif lines_cleared == 3: line_reward = weights['lines_3']
    elif lines_cleared == 4: line_reward = weights['lines_4']

    total_score = (
        line_reward
        - (weights['height'] * height)
        - (weights['holes'] * holes)
        - (weights['bumpiness'] * bumpiness)
        + (weights['wells'] * wells)
    )
    
    return total_score, lines_cleared, height, holes, bumpiness, wells