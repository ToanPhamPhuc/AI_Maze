import pygame
import sys
import random

# Directions: (dy, dx)
DIRS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
KEY_TO_DIR = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
}

CELL_SIZE = 24  # Default, will be recalculated
MIN_SCREEN_W, MIN_SCREEN_H = 400, 400
MAX_SCREEN_W, MAX_SCREEN_H = 1920, 1080
WALL_COLOR = (40, 40, 40)
PATH_COLOR = (220, 220, 220)
PLAYER_COLOR = (0, 120, 255)
EXIT_COLOR = (0, 200, 0)
START_COLOR = (255, 200, 0)
BG_COLOR = (30, 30, 30)

class Maze:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.maze = [['#'] * (2 * width + 1) for _ in range(2 * height + 1)]
        self._generate_maze()
        self.start = (1, 1)
        self.end = (2 * height - 1, 2 * width - 1)
        self.player = list(self.start)

    def _generate_maze(self):
        visited = [[False] * self.width for _ in range(self.height)]
        stack = [(0, 0)]
        while stack:
            y, x = stack[-1]
            visited[y][x] = True
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(dirs)
            found = False
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and not visited[ny][nx]:
                    self.maze[y * 2 + 1 + dy][x * 2 + 1 + dx] = ' '
                    self.maze[ny * 2 + 1][nx * 2 + 1] = ' '
                    stack.append((ny, nx))
                    found = True
                    break
            if not found:
                stack.pop()
        self.maze[1][1] = 'S'
        self.maze[2 * self.height - 1][2 * self.width - 1] = 'E'

    def move_player(self, direction):
        dy, dx = DIRS[direction]
        ny, nx = self.player[0] + dy, self.player[1] + dx
        if 0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0]) and self.maze[ny][nx] != '#':
            self.player = [ny, nx]
            return True
        return False

    def is_finished(self):
        return tuple(self.player) == self.end

def get_maze_size():
    while True:
        try:
            h = int(input('Enter maze height (min 3): '))
            w = int(input('Enter maze width (min 3): '))
            if h >= 3 and w >= 3:
                return h, w
            else:
                print('Please enter values >= 3.')
        except ValueError:
            print('Invalid input. Please enter integers.')

def draw_maze(screen, maze):
    for y, row in enumerate(maze.maze):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell == '#':
                pygame.draw.rect(screen, WALL_COLOR, rect)
            elif cell == 'S':
                pygame.draw.rect(screen, START_COLOR, rect)
            elif cell == 'E':
                pygame.draw.rect(screen, EXIT_COLOR, rect)
            else:
                pygame.draw.rect(screen, PATH_COLOR, rect)
    # Draw player
    py, px = maze.player
    prect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, PLAYER_COLOR, prect)

def main():
    print('--- Maze Game (Pygame) ---')
    h, w = get_maze_size()
    maze = Maze(h, w)
    pygame.init()
    # Calculate cell size and screen size
    maze_pixel_w = (2 * w + 1)
    maze_pixel_h = (2 * h + 1)
    cell_w = MAX_SCREEN_W // maze_pixel_w
    cell_h = MAX_SCREEN_H // maze_pixel_h
    cell_size = min(cell_w, cell_h, 48)  # Cap cell size for very small mazes
    # Ensure minimum screen size
    screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w * cell_size))
    screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h * cell_size))
    global CELL_SIZE
    CELL_SIZE = cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Maze Game')
    clock = pygame.time.Clock()
    finished = False
    move_dir = None
    move_delay = 120  # milliseconds between moves when holding
    last_move_time = 0
    steps = 0
    start_time = pygame.time.get_ticks()
    solve_time = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if not finished:
                if event.type == pygame.KEYDOWN:
                    if event.key in KEY_TO_DIR:
                        move_dir = KEY_TO_DIR[event.key]
                        last_move_time = 0  # move immediately
                    if event.key == pygame.K_r:
                        maze = Maze(h, w)
                        finished = False
                        steps = 0
                        start_time = pygame.time.get_ticks()
                        solve_time = None
                if event.type == pygame.KEYUP:
                    if event.key in KEY_TO_DIR and move_dir == KEY_TO_DIR[event.key]:
                        move_dir = None
            else:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    maze = Maze(h, w)
                    finished = False
                    steps = 0
                    start_time = pygame.time.get_ticks()
                    solve_time = None
        now = pygame.time.get_ticks()
        if not finished and move_dir:
            if last_move_time == 0 or now - last_move_time >= move_delay:
                if maze.move_player(move_dir):
                    last_move_time = now
                    steps += 1
                else:
                    last_move_time = now  # still update to prevent rapid wall bumping
        screen.fill(BG_COLOR)
        draw_maze(screen, maze)
        # Draw stats
        font = pygame.font.SysFont(None, 28)
        elapsed = (solve_time if solve_time is not None else now) - start_time
        elapsed_sec = elapsed // 1000
        steps_surf = font.render(f"Steps: {steps}", True, (255,255,255))
        time_surf = font.render(f"Time: {elapsed_sec}s", True, (255,255,255))
        screen.blit(steps_surf, (10, 10))
        screen.blit(time_surf, (10, 40))
        if finished:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))  # 160/255 alpha for dimming
            screen.blit(overlay, (0, 0))
            big_font = pygame.font.SysFont(None, 48)
            msg1 = big_font.render("Congratulations!", True, (255,255,0))
            msg2 = font.render(f"Solved in {steps} steps, {elapsed_sec} seconds", True, (255,255,255))
            msg3 = font.render("Press R for a new maze", True, (255,255,255))
            screen.blit(msg1, (screen.get_width()//2 - msg1.get_width()//2, screen.get_height()//2 - 60))
            screen.blit(msg2, (screen.get_width()//2 - msg2.get_width()//2, screen.get_height()//2))
            screen.blit(msg3, (screen.get_width()//2 - msg3.get_width()//2, screen.get_height()//2 + 40))
        pygame.display.flip()
        if not finished and maze.is_finished():
            finished = True
            solve_time = pygame.time.get_ticks() - start_time
        clock.tick(60)

if __name__ == '__main__':
    main()
