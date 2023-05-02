





import copy, random, sys

# This game requires a sudokupuzzle.txt file that contains the puzzles.
# Download it from https://inventwithpython.com/sudokupuzzles.txt
# Here's a sample of the content in this file:
# ..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..
# 2...8.3...6..7..84.3.5..2.9...1.54.8.........4.27.6...3.1..7.4.72..4..6...4.1...3
# ......9.7...42.18....7.5.261..9.4....5.....4....5.7..992.1.8....34.59...5.7......
# .3..5..4...8.1.5..46.....12.7.5.2.8....6.3....4.1.9.3.25.....98..1.2.6...8..6..2.

# Set up the constants:
EMPTY_SPACE = '.'
GRID_LENGTH = 9
BOX_LENGTH = 3
FULL_GRID_SIZE = GRID_LENGTH * GRID_LENGTH


class SudokuGrid:
    def __init__(self, originalSetup):
        # originalSetup is a string of 81 characters for the puzzle
        # setup, with numbers and periods (for the blank spaces).
        # See https://inventwithpython.com/sudokupuzzles.txt
        self.originalSetup = originalSetup

        # The state of the sudoku grid is represented by a dictionary
        # with (x, y) keys and values of the number (as a string) at
        # that space.
        self.grid = {}
        self.resetGrid()  # Set the grid state to its original setup.
        self.moves = []  # Tracks each move for the undo feature.

    def resetGrid(self):
        """Reset the state of the grid, tracked by self.grid, to the
        state in self.originalSetup."""
        for x in range(1, GRID_LENGTH + 1):
            for y in range(1, GRID_LENGTH + 1):
                self.grid[(x, y)] = EMPTY_SPACE

        assert len(self.originalSetup) == FULL_GRID_SIZE
        i = 0  # i goes from 0 to 80
        y = 0  # y goes from 0 to 8
        while i < FULL_GRID_SIZE:
            for x in range(GRID_LENGTH):
                