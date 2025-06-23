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
MIN_SCREEN_W, MIN_SCREEN_H = 1366, 768
MAX_SCREEN_W, MAX_SCREEN_H = 1536, 864

WALL_COLOR = (40, 40, 40)
PATH_COLOR = (220, 220, 220)
PLAYER_COLOR = (0, 120, 255)
EXIT_COLOR = (0, 200, 0)
START_COLOR = (255, 200, 0)
BG_COLOR = (30, 30, 30)
PLAYER_TRAIL_COLOR = (0, 120, 255, 80)  # RGBA for semi-transparent trail

class Maze:
    def __init__(self, height, width, cell_size):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.maze = [['#'] * (2 * width + 1) for _ in range(2 * height + 1)]
        self._generate_maze()
        self.start = (1, 1)
        self.end = (2 * height - 1, 2 * width - 1)
        self.player = list(self.start)
        self.pixel_w = len(self.maze[0]) * self.cell_size
        self.pixel_h = len(self.maze) * self.cell_size

    def _generate_maze(self):
        visited = [[False] * self.width for _ in range(self.height)]
        stack = [(0, 0)]
        while stack:
            y, x = stack[-1]
            visited[y][x] = True
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(dirs)
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and not visited[ny][nx]:
                    self.maze[y * 2 + 1 + dy][x * 2 + 1 + dx] = ' '
                    self.maze[ny * 2 + 1][nx * 2 + 1] = ' '
                    stack.append((ny, nx))
                    break
            else:
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

def draw_maze(screen, maze, trail=None):
    offset_x = (screen.get_width() - maze.pixel_w) // 2 if screen.get_width() > maze.pixel_w else 0
    offset_y = (screen.get_height() - maze.pixel_h) // 2 if screen.get_height() > maze.pixel_h else 0
    for y, row in enumerate(maze.maze):
        for x, cell in enumerate(row):
            rect = pygame.Rect(offset_x + x * maze.cell_size, offset_y + y * maze.cell_size, maze.cell_size, maze.cell_size)
            if cell == '#':
                pygame.draw.rect(screen, WALL_COLOR, rect)
            elif cell == 'S':
                pygame.draw.rect(screen, START_COLOR, rect)
            elif cell == 'E':
                pygame.draw.rect(screen, EXIT_COLOR, rect)
            else:
                pygame.draw.rect(screen, PATH_COLOR, rect)
    # Draw trail if enabled
    if trail:
        trail_surf = pygame.Surface((maze.cell_size, maze.cell_size), pygame.SRCALPHA)
        trail_surf.fill(PLAYER_TRAIL_COLOR)
        for (ty, tx) in trail:
            screen.blit(trail_surf, (offset_x + tx * maze.cell_size, offset_y + ty * maze.cell_size))
    # Draw player
    py, px = maze.player
    prect = pygame.Rect(offset_x + px * maze.cell_size, offset_y + py * maze.cell_size, maze.cell_size, maze.cell_size)
    pygame.draw.rect(screen, PLAYER_COLOR, prect)

def reset_maze(h, w, cell_size):
    maze = Maze(h, w, cell_size)
    finished = False
    steps = 0
    start_time = pygame.time.get_ticks()
    solve_time = None
    trail = set()
    return maze, finished, steps, start_time, solve_time, trail

def draw_menu(screen, selected_idx):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    title = font.render("Select Difficulty", True, (255,255,0))
    options = [
        "Beginner (8x8)",
        "Intermediate (16x16)",
        "Expert (30x16)",
        "Custom"
    ]
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
    for i, opt in enumerate(options):
        color = (255,255,255) if i == selected_idx else (180,180,180)
        surf = small_font.render(opt, True, color)
        screen.blit(surf, (screen.get_width()//2 - surf.get_width()//2, 200 + i*60))
    pygame.display.flip()

def draw_custom_input(screen, width_str, height_str, active_field):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    title = font.render("Custom Maze Size", True, (255,255,0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
    w_label = small_font.render("Width:", True, (255,255,255))
    h_label = small_font.render("Height:", True, (255,255,255))
    w_color = (0,255,0) if active_field == 'width' else (255,255,255)
    h_color = (0,255,0) if active_field == 'height' else (255,255,255)
    w_input = small_font.render(width_str or "_", True, w_color)
    h_input = small_font.render(height_str or "_", True, h_color)
    screen.blit(w_label, (screen.get_width()//2 - 120, 200))
    screen.blit(w_input, (screen.get_width()//2, 200))
    screen.blit(h_label, (screen.get_width()//2 - 120, 260))
    screen.blit(h_input, (screen.get_width()//2, 260))
    instr = small_font.render("Enter numbers, press Enter to confirm", True, (180,180,180))
    screen.blit(instr, (screen.get_width()//2 - instr.get_width()//2, 340))
    pygame.display.flip()

def main():
    print('--- Maze Game (Pygame) ---')
    pygame.init()
    screen = pygame.display.set_mode((MIN_SCREEN_W, MIN_SCREEN_H))
    pygame.display.set_caption('Maze Game')
    clock = pygame.time.Clock()
    # Menu state
    menu_state = 'menu'  # 'menu', 'custom', 'game'
    selected_idx = 0
    width_str = ''
    height_str = ''
    active_field = 'width'
    h, w = 8, 8  # default
    cell_size = 24
    maze = None
    finished = False
    move_dir = None
    move_delay = 120
    last_move_time = 0
    show_trail = False
    steps = 0
    start_time = 0
    solve_time = None
    trail = set()
    while True:
        if menu_state == 'menu':
            draw_menu(screen, selected_idx)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_idx = (selected_idx - 1) % 4
                    elif event.key == pygame.K_DOWN:
                        selected_idx = (selected_idx + 1) % 4
                    elif event.key == pygame.K_RETURN:
                        if selected_idx == 0:
                            h, w = 8, 8
                            menu_state = 'game'
                        elif selected_idx == 1:
                            h, w = 16, 16
                            menu_state = 'game'
                        elif selected_idx == 2:
                            h, w = 16, 30
                            menu_state = 'game'
                        elif selected_idx == 3:
                            menu_state = 'custom'
                            width_str = ''
                            height_str = ''
                            active_field = 'width'
        elif menu_state == 'custom':
            draw_custom_input(screen, width_str, height_str, active_field)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        active_field = 'height' if active_field == 'width' else 'width'
                    elif event.key == pygame.K_RETURN:
                        try:
                            w = int(width_str)
                            h = int(height_str)
                            if w >= 3 and h >= 3:
                                menu_state = 'game'
                            else:
                                width_str = ''
                                height_str = ''
                        except:
                            width_str = ''
                            height_str = ''
                    elif event.key == pygame.K_BACKSPACE:
                        if active_field == 'width':
                            width_str = width_str[:-1]
                        else:
                            height_str = height_str[:-1]
                    elif event.unicode.isdigit():
                        if active_field == 'width':
                            width_str += event.unicode
                        else:
                            height_str += event.unicode
        elif menu_state == 'game':
            # Calculate cell size and screen size
            maze_pixel_w = (2 * w + 1)
            maze_pixel_h = (2 * h + 1)
            cell_w = MAX_SCREEN_W // maze_pixel_w
            cell_h = MAX_SCREEN_H // maze_pixel_h
            cell_size = min(cell_w, cell_h, 48)
            screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w * cell_size))
            screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h * cell_size))
            screen = pygame.display.set_mode((screen_width, screen_height))
            maze, finished, steps, start_time, solve_time, trail = reset_maze(h, w, cell_size)
            menu_state = 'playing'
            move_dir = None
            last_move_time = 0
            show_trail = False
        elif menu_state == 'playing':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in KEY_TO_DIR:
                        move_dir = KEY_TO_DIR[event.key]
                        last_move_time = 0
                    elif event.key == pygame.K_r:
                        menu_state = 'menu'
                        selected_idx = 0
                    elif event.key == pygame.K_t:
                        show_trail = not show_trail
                if event.type == pygame.KEYUP:
                    if event.key in KEY_TO_DIR and move_dir == KEY_TO_DIR[event.key]:
                        move_dir = None
            now = pygame.time.get_ticks()
            if not finished and move_dir:
                if last_move_time == 0 or now - last_move_time >= move_delay:
                    moved = maze.move_player(move_dir)
                    last_move_time = now
                    if moved:
                        steps += 1
                        trail.add(tuple(maze.player))
            screen.fill(BG_COLOR)
            draw_maze(screen, maze, trail if show_trail else None)
            font = pygame.font.SysFont(None, 28)
            elapsed = (solve_time if solve_time is not None else now) - start_time
            elapsed_sec = elapsed // 1000
            steps_surf = font.render(f"Steps: {steps}", True, (255,255,255))
            time_surf = font.render(f"Time: {elapsed_sec}s", True, (255,255,255))
            screen.blit(steps_surf, (10, 10))
            screen.blit(time_surf, (10, 40))
            hint_alpha = 80 if not show_trail else 200
            trail_hint_surf = font.render("Press T to show trail", True, (180,180,180))
            trail_hint_surf = trail_hint_surf.convert_alpha()
            trail_hint_surf.set_alpha(hint_alpha)
            screen.blit(trail_hint_surf, (10, 70))
            if finished:
                overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 160))
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
