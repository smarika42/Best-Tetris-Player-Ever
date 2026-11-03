def get_column_heights(board):
    rows = len(board)
    cols = len(board[0])
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if board[r][c] == 1:
                heights[c] = rows - r
                break
    return heights


def calculate_board_height(board):
    heights = get_column_heights(board)
    # Penalize total height across the board PLUS extra penalty for the tallest tower
    # so the AI never builds a single skyscraper on the left side!
    main_heights = heights[:-1]
    tower_span = max(main_heights) - min(main_heights)
    return sum(heights) + (max(heights) * 2.5) + (tower_span * 1.5)


def count_holes(board):
    """
    Counts empty cells buried under blocks, plus a small 'roof' penalty for blocks
    sitting above holes (so the AI doesn't bury a hole 8 blocks deep!).
    """
    total_holes = 0.0
    rows = len(board)
    cols = len(board[0])

    for col in range(cols):
        block_found = False
        blocks_above = 0
        for row in range(rows):
            if board[row][col] == 1:
                block_found = True
                blocks_above += 1
            elif board[row][col] == 0 and block_found:
                # Base hole penalty + slight penalty for burying it deeper
                total_holes += 1.0 + min(blocks_above, 4) * 0.15
    return total_holes


def calculate_bumpiness(board):
    heights = get_column_heights(board)
    cols = len(heights)

    # Measure bumpiness across the 9 main columns (Columns 0 to 8)
    # Square large jumps (steps >= 3) so it never builds cliffs that only an 'I' piece can fix!
    total_bumpiness = 0.0
    for i in range(cols - 2):
        diff = abs(heights[i] - heights[i + 1])
        if diff >= 3:
            total_bumpiness += diff * 1.8
        else:
            total_bumpiness += diff

    # Heavy penalty if Column 9 (our Tetris well!) becomes taller than Column 8
    if heights[cols - 1] > heights[cols - 2]:
        total_bumpiness += (heights[cols - 1] - heights[cols - 2]) * 4.0

    return total_bumpiness


def calculate_wells(board):
    """
    Rewards building a clean, flat 9-column stack (Cols 0..8) with Column 9 completely open!
    Previously, `min(heights[:9])` returned 0 until all 9 columns had blocks, which caused
    the AI to dump early pieces into Column 9!
    """
    heights = get_column_heights(board)
    cols = len(heights)
    main_stack = heights[:cols - 1]
    well_col_height = heights[cols - 1]

    # Use average and median-like floor of the main stack so keeping Col 9 empty is rewarded from Piece #1!
    avg_main = sum(main_stack) / float(len(main_stack))
    min_main = min(main_stack)

    # Penalize putting ANY blocks into Column 9 while building the stack
    if well_col_height > 0:
        well_penalty = -3.5 * well_col_height
    else:
        well_penalty = 2.0  # Bonus for keeping Column 9 at 0!

    # Reward both true Tetris-ready depth (min_main - well_col_height) and partial stack depth
    true_depth = max(0.0, float(min_main - well_col_height))
    developing_depth = max(0.0, avg_main - well_col_height) * 0.5

    return min(true_depth, 4.0) + min(developing_depth, 2.5) + well_penalty


def evaluate_board(board, lines_cleared, weights):
    height = calculate_board_height(board)
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(board)
    wells = calculate_wells(board)

    line_reward = 0.0
    if lines_cleared == 1:
        line_reward = weights['lines_1']
    elif lines_cleared == 2:
        line_reward = weights['lines_2']
    elif lines_cleared == 3:
        line_reward = weights['lines_3']
    elif lines_cleared == 4:
        line_reward = weights['lines_4']

    total_score = (
        line_reward
        - (weights['height'] * height)
        - (weights['holes'] * holes)
        - (weights['bumpiness'] * bumpiness)
        + (weights['wells'] * wells)
    )

    return total_score, lines_cleared, height, holes, bumpiness, wells
