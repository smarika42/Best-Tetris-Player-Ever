Tetris AI Bot
Ever wish you had a super-smart friend sitting next to you, telling you the exact best move in Tetris before your block even hits the ground? That’s what this project does!

It uses a webcam/camera to watch your screen while you play, figures out which block is falling, calculates the smartest spot for it, and prints the exact steps to win.

How It Works in 3 Simple Steps
SEE (The Eyes)
The webcam/camera looks at the phone screen. The program checks the bright colors to instantly spot which piece is coming up next, whether it's the long cyan line, the yellow square, or the red Z-block.

THINK (The Brain)
In less than a single millisecond, the AI tests every possible place to drop the block. It scores each option based on four simple rules:
Keep it low: Don't let the blocks stack too high.
Clear lines: Bonus points for clearing full rows!
No empty holes: Avoid trapping empty spaces underneath blocks.
Keep it flat: Avoid making the top surface super bumpy.

TELL (The Voice)
The program prints out simple, step-by-step instructions on screen right away, like:
[J] DO ALL: ROTATE -> MOVE LEFT -> MOVE LEFT -> DROP

What You Need
A computer with a webcam
A screen playing Tetris
Python

How to Play
Download the code onto your computer.
Install the extra tools by opening your command line/terminal and typing:

pip install opencv-python numpy
python main.py

Point your webcam at the screen so the game fits inside the blue box on screen. Follow the prompts and beat your high score!

Pro Tips for Best Results
Turn up your brightness: The camera needs to see bright colors clearly.
Turn off Night Mode / Blue Light Filter: Warm color filters confuse the bot when it tries to tell the difference between red, orange, and yellow blocks.
Avoid glare: Try not to point bright overhead lights directly at your phone screen.
