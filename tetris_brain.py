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
    main_heights = heights[:-1]
    # Penalize overall height + highest column + height difference across the 9 main columns
    tower_span = max(main_heights) - min(main_heights)
    return sum(heights) + (max(heights) * 2.5) + (tower_span * 1.8)


def count_holes(board):
    """
    Counts empty cells buried under blocks, plus a roof penalty for every block sitting
    above a hole so the AI actively uncovers holes instead of burying them deeper!
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
                total_holes += 1.0 + (min(blocks_above, 5) * 0.20)
    return total_holes


def calculate_bumpiness(board):
    heights = get_column_heights(board)
    cols = len(heights)

    total_bumpiness = 0.0
    # Measure bumpiness across the 9 main stack columns (Cols 0..8)
    for i in range(cols - 2):
        diff = abs(heights[i] - heights[i + 1])
        # Penalize deep 2+ block canyons between columns 0..8 so it doesn't build narrow wells in the middle!
        if diff >= 2:
            total_bumpiness += diff * 1.75
        else:
            total_bumpiness += diff

    # Extra penalty if an internal column (1..7) is a deep canyon surrounded by taller walls on both sides
    for i in range(1, cols - 2):
        wall_min = min(heights[i - 1], heights[i + 1])
        if wall_min - heights[i] >= 2:
            total_bumpiness += (wall_min - heights[i]) * 1.5

    # Heavy penalty if Column 9 (the right-hand Tetris well) is taller than Column 8
    if heights[cols - 1] > heights[cols - 2]:
        total_bumpiness += (heights[cols - 1] - heights[cols - 2]) * 4.0

    return total_bumpiness


def calculate_wells(board):
    """
    Rewards building a flat 9-column stack (Cols 0..8) while keeping Column 9 open for an 'I' bar!
    """
    heights = get_column_heights(board)
    cols = len(heights)
    main_stack = heights[:cols - 1]
    well_col_height = heights[cols - 1]

    avg_main = sum(main_stack) / float(len(main_stack))
    min_main = min(main_stack)

    # Strong penalty for dropping ANY block into Column 9 unless it clears lines
    if well_col_height > 0:
        well_penalty = -4.5 * well_col_height
    else:
        well_penalty = 2.5

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