
def calculate_board_height(board):
    total_height = 0
    rows = len(board)
    cols = len(board[0])

    for col in range(cols):
        for row in range(rows):
            if board[row][col] == 1:
                col_height = rows - row
                total_height += col_height
                break
    return total_height


def count_holes(board):
    total_holes = 0
    rows = len(board)
    cols = len(board[0])

    for col in range(cols):
        block_found = False
        for row in range(rows):
            if board[row][col] ==1:
                block_found = True
            elif board[row][col] == 0 and block_found:
                total_holes += 1
    return total_holes

def calculate_bumpiness(board):
    rows = len(board)
    cols = len(board[0])
    col_heights = []

    # 1. Figure out the height of each individual column
    for col in range(cols):
        current_height = 0
        for row in range(rows):
            if board[row][col] == 1:
                current_height = rows - row
                break # Found the top block, stop checking this column
        col_heights.append(current_height)

    # 2. Calculate the difference between adjacent columns
    total_bumpiness = 0
    for i in range(len(col_heights) - 1):
        # abs() makes sure a difference of -2 becomes just 2
        difference = abs(col_heights[i] - col_heights[i + 1])
        total_bumpiness += difference

    return total_bumpiness

def count_completed_lines(board):
    completed_lines = 0
    cols = len(board[0])
    
    # Check each row directly
    for row in board:
        # If the row has all 1s, the sum of the row will equal the number of columns
        if sum(row) == cols:
            completed_lines += 1
            
    return completed_lines

def calculate_wells(board):
    wells = 0
    rows = len(board)
    cols = len(board[0])

    for col in range(cols):
        for row in range(rows):
            # If the space is empty...
            if board[row][col] == 0:
                # Check if the left side is a wall (col == 0) or a solid block
                left_solid = (col == 0) or (board[row][col - 1] == 1)
                
                # Check if the right side is a wall (col == cols - 1) or a solid block
                right_solid = (col == cols - 1) or (board[row][col + 1] == 1)
                
                if left_solid and right_solid:
                    # Make sure there is NO block above it blocking the well!
                    open_to_sky = True
                    for r in range(row):
                        if board[r][col] == 1:
                            open_to_sky = False
                            break
                            
                    if open_to_sky:
                        wells += 1 # We found a perfect well space!
                        
    return wells

# Notice we added 'lines_cleared' right here in the parentheses!
def evaluate_board(board, lines_cleared, weights):
    height = calculate_board_height(board)
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(board)
    wells = calculate_wells(board) # 1. Run the new well scanner!

    line_reward = 0
    if lines_cleared == 1: line_reward = weights['lines_1']
    elif lines_cleared == 2: line_reward = weights['lines_2']
    elif lines_cleared == 3: line_reward = weights['lines_3']
    elif lines_cleared == 4: line_reward = weights['lines_4']

    # 2. Add the wells weight into the final score!
    total_score = (line_reward) - (weights['height'] * height) - (weights['holes'] * holes) - (weights['bumpiness'] * bumpiness) + (weights['wells'] * wells)
    
    # 3. Return the 6 values (added wells at the end)
    return total_score, lines_cleared, height, holes, bumpiness, wells