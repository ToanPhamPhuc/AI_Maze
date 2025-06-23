# Maze Game Configuration

# Screen size limits
MIN_SCREEN_W, MIN_SCREEN_H = 1366, 768
MAX_SCREEN_W, MAX_SCREEN_H = 1920, 1080

# Cell size
DEFAULT_CELL_SIZE = 24

# Colors
WALL_COLOR = (40, 40, 40)
PATH_COLOR = (220, 220, 220)
PLAYER_COLOR = (0, 120, 255)
EXIT_COLOR = (0, 200, 0)
START_COLOR = (255, 200, 0)
BG_COLOR = (30, 30, 30)
PLAYER_TRAIL_COLOR = (0, 120, 255, 80)  # RGBA for semi-transparent trail

# Trail
TRAIL_HINT_COLOR = (255, 0, 0)

# Movement
MOVE_DELAY = 120  # milliseconds between moves when holding

# High score directory
SCORES_DIR = 'scores'

# Menu options
MENU_OPTIONS = [
    "Beginner (8x8)",
    "Intermediate (16x16)",
    "Expert (30x16)",
    "Custom",
    "Quit"
] 