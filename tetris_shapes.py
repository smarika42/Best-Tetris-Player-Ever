# A dictionary containing the 7 standard Tetris pieces (Tetrominoes)
# 1 represents a block, 0 represents empty space in the piece's bounding box.

SHAPES = {
    'I': [
        [1, 1, 1, 1]
    ],
    'O': [
        [1, 1],
        [1, 1]
    ],
    'T': [
        [0, 1, 0],
        [1, 1, 1]
    ],
    'S': [
        [0, 1, 1],
        [1, 1, 0]
    ],
    'Z': [
        [1, 1, 0],
        [0, 1, 1]
    ],
    'J': [
        [1, 0, 0],
        [1, 1, 1]
    ],
    'L': [
        [0, 0, 1],
        [1, 1, 1]
    ]
}
def rotate_piece(piece_matrix):
    # This mathematical trick rotates any 2D list 90 degrees clockwise!
    # It pairs up the columns, reverses them, and turns them into new rows.
    rotated = [list(row) for row in zip(*piece_matrix[::-1])]
    return rotated

