import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pygame
import time
import threading
from collections import deque
import heapq
from Dijkstra.scripts.environment import MazeEnvironment
from GAME.ui.menu import draw_menu
from GAME.ui.custom_input import draw_custom_input
from GAME.configs.config import MENU_OPTIONS, SCORES_DIR, MIN_SCREEN_W, MIN_SCREEN_H, DEFAULT_CELL_SIZE

# Map menu options to maze sizes
DIFFICULTY_MAP = {
    0: (3, 3),   # Beginner
    1: (8, 8),   # Intermediate
    2: (16, 16), # Expert
}

# Algorithm implementations
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def solve_dijkstra(env, on_expand=None, delay=0, screen=None):
    """Dijkstra's algorithm implementation"""
    start = tuple(env.maze.player)
    goal = tuple(env.maze.end)
    maze = env.maze.maze
    h, w = len(maze), len(maze[0])
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add(start)
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while queue:
        (y, x), path = queue.popleft()
        if (y, x) == goal:
            env.solution_path = path
            return path
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != '#':
                npos = (ny, nx)
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, path + [npos]))
                    if on_expand:
                        on_expand(npos)
                    if delay > 0 and screen is not None:
                        env.render(screen, show_solution=False, highlight=npos)
                        pygame.time.delay(delay)
    env.solution_path = []
    return []

def solve_astar(env, on_expand=None, delay=0, screen=None):
    """A* algorithm implementation"""
    start = tuple(env.maze.player)
    goal = tuple(env.maze.end)
    maze = env.maze.maze
    h, w = len(maze), len(maze[0])
    visited = set()
    queue = [(manhattan_distance(start, goal), 0, start, [start])]
    visited.add(start)
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while queue:
        f_score, g_score, (y, x), path = heapq.heappop(queue)
        if (y, x) == goal:
            env.solution_path = path
            return path
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != '#':
                npos = (ny, nx)
                if npos not in visited:
                    visited.add(npos)
                    new_g = g_score + 1
                    new_f = new_g + manhattan_distance(npos, goal)
                    new_path = path + [npos]
                    heapq.heappush(queue, (new_f, new_g, npos, new_path))
                    if on_expand:
                        on_expand(npos)
                    if delay > 0 and screen is not None:
                        env.render(screen, show_solution=False, highlight=npos)
                        pygame.time.delay(delay)
    env.solution_path = []
    return []

def solve_dfs(env, on_expand=None, delay=0, screen=None):
    """DFS algorithm implementation"""
    start = tuple(env.maze.player)
    goal = tuple(env.maze.end)
    maze = env.maze.maze
    h, w = len(maze), len(maze[0])
    visited = set()
    stack = [(start, [start])]
    visited.add(start)
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while stack:
        (y, x), path = stack.pop()
        if (y, x) == goal:
            env.solution_path = path
            return path
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != '#':
                npos = (ny, nx)
                if npos not in visited:
                    visited.add(npos)
                    stack.append((npos, path + [npos]))
                    if on_expand:
                        on_expand(npos)
                    if delay > 0 and screen is not None:
                        env.render(screen, show_solution=False, highlight=npos)
                        pygame.time.delay(delay)
    env.solution_path = []
    return []

def solve_bfs(env, on_expand=None, delay=0, screen=None):
    """BFS algorithm implementation"""
    start = tuple(env.maze.player)
    goal = tuple(env.maze.end)
    maze = env.maze.maze
    h, w = len(maze), len(maze[0])
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add(start)
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while queue:
        (y, x), path = queue.popleft()
        if (y, x) == goal:
            env.solution_path = path
            return path
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != '#':
                npos = (ny, nx)
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, path + [npos]))
                    if on_expand:
                        on_expand(npos)
                    if delay > 0 and screen is not None:
                        env.render(screen, show_solution=False, highlight=npos)
                        pygame.time.delay(delay)
    env.solution_path = []
    return []

class BotRunner:
    def __init__(self, name, solve_func, color):
        self.name = name
        self.solve_func = solve_func
        self.color = color
        self.env = None
        self.solution_path = []
        self.expanded = set()  # Each bot has its own expanded set
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.expanded_count = 0
        self.screen = None
        self.x_offset = 0
        self.y_offset = 0
        self.cell_size = 0
        self.is_running = False
        self.last_render_time = 0
        self.render_interval = 50  # Render every 50ms

    def reset(self, width, height, screen=None, x_offset=0, y_offset=0, cell_size=0):
        self.env = MazeEnvironment(height, width, DEFAULT_CELL_SIZE)
        self.env.reset()
        self.solution_path = []
        self.expanded = set()  # Reset bot's own expanded set
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.expanded_count = 0
        self.screen = screen
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.cell_size = cell_size
        self.is_running = False
        self.last_render_time = 0

    def on_expand(self, pos):
        """Callback for when a node is expanded - for animation"""
        if self.screen and self.is_running:
            # Add to bot's own expanded set
            self.expanded.add(pos)
            # Only render periodically to avoid lag
            current_time = pygame.time.get_ticks()
            if current_time - self.last_render_time > self.render_interval:
                self.render_bot()
                self.last_render_time = current_time
            time.sleep(0.02)  # Slow down the search for visible animation

    def render_bot(self):
        """Render this bot's maze"""
        if not self.env or not self.screen:
            return
        
        # Draw maze
        maze = self.env.maze.maze
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                rect = pygame.Rect(self.x_offset + x * self.cell_size, self.y_offset + y * self.cell_size, self.cell_size, self.cell_size)
                cell = maze[y][x]
                if cell == '#':
                    pygame.draw.rect(self.screen, (50, 50, 50), rect)  # Wall
                elif cell == 'S':
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)   # Start
                elif cell == 'E':
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)   # End
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)  # Path
        
        # Draw expanded nodes using bot's own expanded set
        for (ey, ex) in self.expanded:
            rect = pygame.Rect(self.x_offset + ex * self.cell_size, self.y_offset + ey * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (100, 150, 255), rect)
        
        # Draw solution path
        if self.solution_path:
            for (py, px) in self.solution_path:
                rect = pygame.Rect(self.x_offset + px * self.cell_size, self.y_offset + py * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.color, rect, 2)
        
        # Draw player
        py, px = self.env.maze.player
        prect = pygame.Rect(self.x_offset + px * self.cell_size, self.y_offset + py * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 255, 0), prect)
        
        # Draw border
        pygame.draw.rect(self.screen, self.color, (self.x_offset, self.y_offset, len(maze[0]) * self.cell_size, len(maze) * self.cell_size), 3)

    def run(self):
        self.is_running = True
        self.start_time = time.time()
        self.solution_path = self.solve_func(self.env, on_expand=self.on_expand)
        self.end_time = time.time()
        self.steps = len(self.solution_path) - 1 if self.solution_path else 0
        self.expanded_count = len(self.expanded)  # Use bot's own expanded set
        self.finished = True
        self.is_running = False
        # Final render to show complete result
        if self.screen:
            self.render_bot()

def format_time(seconds):
    if seconds is None:
        return "N/A"
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return '{}m {}s {}ms'.format(m, s, ms)

def render_status(screen, bot, x_offset, y_offset, cell_size):
    """Render bot status information"""
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    
    # Bot name
    name_text = font.render(bot.name, True, bot.color)
    screen.blit(name_text, (x_offset, y_offset - 30))
    
    if bot.finished:
        # Status information TODO
        # elapsed = bot.end_time - bot.start_time if bot.start_time and bot.end_time else None
        # time_text = small_font.render(f'Time: {format_time(elapsed)}', True, (255, 255, 255))
        # steps_text = small_font.render(f'Steps: {bot.steps}', True, (255, 255, 255))
        # expanded_text = small_font.render(f'Expanded: {bot.expanded_count}', True, (255, 255, 255))
        
        # screen.blit(time_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 5))
        # screen.blit(steps_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 20))
        # screen.blit(expanded_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 35))
        pass
    else:
        # Running status
        status_text = small_font.render('Running...', True, (255, 255, 0))
        screen.blit(status_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 5))

def render_comparison_table(screen, bots, x_offset, y_offset):
    """Render the comparison table on the right side"""
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    
    # Table title
    title_text = font.render('Algorithm Comparison', True, (255, 255, 255))
    screen.blit(title_text, (x_offset, y_offset))
    
    # Table headers
    headers = ['Algorithm', 'Distance', 'Expanded', 'Place', 'Time']
    header_y = y_offset + 40
    
    for i, header in enumerate(headers):
        header_text = small_font.render(header, True, (200, 200, 200))
        screen.blit(header_text, (x_offset + i * 80, header_y))
    
    # Calculate places based on completion time
    finished_bots = [bot for bot in bots if bot.finished]
    if finished_bots:
        # Sort by completion time (faster = better place)
        finished_bots.sort(key=lambda b: b.end_time - b.start_time if b.start_time and b.end_time else float('inf'))
        
        # Assign places
        for i, bot in enumerate(finished_bots):
            bot.place = i + 1
    
    # Table data
    data_y = header_y + 30
    for bot in bots:
        # Algorithm name
        name_text = small_font.render(bot.name, True, bot.color)
        screen.blit(name_text, (x_offset, data_y))
        
        if bot.finished:
            # Distance
            distance_text = small_font.render(str(bot.steps), True, (255, 255, 255))
            screen.blit(distance_text, (x_offset + 80, data_y))
            
            # Expanded
            expanded_text = small_font.render(str(bot.expanded_count), True, (255, 255, 255))
            screen.blit(expanded_text, (x_offset + 160, data_y))
            
            # Place
            place_text = small_font.render(str(getattr(bot, 'place', '-')), True, (255, 255, 255))
            screen.blit(place_text, (x_offset + 240, data_y))
            
            # Time
            elapsed = bot.end_time - bot.start_time if bot.start_time and bot.end_time else None
            time_text = small_font.render(format_time(elapsed), True, (255, 255, 255))
            screen.blit(time_text, (x_offset + 320, data_y))
        else:
            # Show "Running..." for unfinished bots
            running_text = small_font.render('Running...', True, (255, 255, 0))
            screen.blit(running_text, (x_offset + 80, data_y))
        
        data_y += 25

def render_controls(screen, x_offset, y_offset):
    """Render the controls section"""
    font = pygame.font.SysFont(None, 20)
    small_font = pygame.font.SysFont(None, 16)
    
    # Controls title
    title_text = font.render('Controls', True, (255, 255, 255))
    screen.blit(title_text, (x_offset, y_offset))
    
    # Control instructions
    controls = [
        'R - Generate new maze',
        'ESC - Back to menu',
        'Q - Quit game'
    ]
    
    for i, control in enumerate(controls):
        control_text = small_font.render(control, True, (200, 200, 200))
        screen.blit(control_text, (x_offset, y_offset + 30 + i * 20))

def run_comparison(screen, width, height):
    """Run the comparison with the specified maze size"""
    # Create bots
    bots = [
        BotRunner("Dijkstra", solve_dijkstra, (0, 255, 0)),      # Green
        BotRunner("A*", solve_astar, (255, 0, 255)),             # Magenta
        BotRunner("DFS", solve_dfs, (0, 255, 255)),              # Cyan
        BotRunner("BFS", solve_bfs, (255, 255, 0))               # Yellow
    ]
    
    # Calculate max allowed grid area (2/3 of 1366x768, but max height 750px)
    max_grid_width = int(1366 * 2 / 3)
    max_grid_height = 750  # Limit height to 750px
    maze_size = width
    maze_pixel_size = maze_size * 2 + 1  # Account for wall cells
    # Compute cell size so that 2x2 grid fits in max area
    cell_size_w = max_grid_width // (2 * maze_pixel_size)
    cell_size_h = max_grid_height // (2 * maze_pixel_size)
    cell_size = max(2, min(cell_size_w, cell_size_h))  # Minimum cell size 2px for visibility
    total_maze_size = maze_pixel_size * cell_size
    
    # Layout calculations
    left_width = total_maze_size * 2 + 50  # 2 mazes wide + padding
    right_width = 400  # Space for table and controls
    total_width = left_width + right_width
    screen_width = max(1366, total_width)
    screen_height = max(768, total_maze_size * 2 + 100)  # 2 mazes high + status area
    
    # Resize screen if needed
    if screen.get_size() != (screen_width, screen_height):
        screen = pygame.display.set_mode((screen_width, screen_height))
    
    clock = pygame.time.Clock()
    
    # Calculate positions for the mazes (left side)
    start_x = 25
    start_y = 50
    
    # Create ONE shared maze environment for all bots
    shared_env = MazeEnvironment(height, width, DEFAULT_CELL_SIZE)
    shared_env.reset()
    
    # Initialize bots with the SAME maze
    positions = [
        (start_x, start_y),                                    # Top-left: Dijkstra
        (start_x + total_maze_size, start_y),                  # Top-right: A*
        (start_x, start_y + total_maze_size),                  # Bottom-left: DFS
        (start_x + total_maze_size, start_y + total_maze_size) # Bottom-right: BFS
    ]
    
    for i, bot in enumerate(bots):
        bot.reset(maze_size, maze_size, screen, positions[i][0], positions[i][1], cell_size)
        # Set all bots to use the same maze environment
        bot.env = shared_env
    
    # Start all bots in separate threads
    threads = []
    for bot in bots:
        thread = threading.Thread(target=bot.run)
        thread.start()
        threads.append(thread)
    
    running = True
    last_update_time = 0
    update_interval = 16  # ~60 FPS updates
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    # Return to menu
                    for thread in threads:
                        thread.join()
                    return screen
                elif event.key == pygame.K_r:  # Restart
                    # Wait for current threads to finish
                    for thread in threads:
                        thread.join()
                    
                    # Create a new shared maze environment
                    shared_env = MazeEnvironment(height, width, DEFAULT_CELL_SIZE)
                    shared_env.reset()
                    
                    # Reset and restart all bots with the new shared maze
                    for i, bot in enumerate(bots):
                        bot.reset(maze_size, maze_size, screen, positions[i][0], positions[i][1], cell_size)
                        bot.env = shared_env
                    
                    # Start new threads
                    threads = []
                    for bot in bots:
                        thread = threading.Thread(target=bot.run)
                        thread.start()
                        threads.append(thread)
        
        # Update screen at consistent intervals
        if current_time - last_update_time > update_interval:
            # Clear screen
            screen.fill((30, 30, 30))
            
            # Render all bots (left side)
            for i, bot in enumerate(bots):
                bot.render_bot()
                render_status(screen, bot, positions[i][0], positions[i][1], cell_size)
            
            # Render comparison table (right side)
            table_x = left_width + 25
            table_y = 50
            render_comparison_table(screen, bots, table_x, table_y)
            
            # Render controls (right side, below table)
            controls_x = left_width + 25
            controls_y = table_y + 200
            render_controls(screen, controls_x, controls_y)
            
            # Draw title
            title_font = pygame.font.SysFont(None, 36)
            title_text = title_font.render('Pathfinding Algorithms Comparison', True, (255, 255, 255))
            screen.blit(title_text, (screen_width//2 - title_text.get_width()//2, 10))
            
            pygame.display.flip()
            last_update_time = current_time
        
        clock.tick(60)  # Cap at 60 FPS
    
    # Wait for threads to finish
    for thread in threads:
        thread.join()
    
    return screen

def main():
    pygame.init()
    screen = pygame.display.set_mode((MIN_SCREEN_W, MIN_SCREEN_H))
    pygame.display.set_caption('Pathfinding Algorithms Comparison')
    clock = pygame.time.Clock()
    running = True
    selected_idx = 0
    
    while running:
        option_rects = draw_menu(screen, selected_idx, highscores={
            'Beginner': (None, None),
            'Intermediate': (None, None),
            'Expert': (None, None),
        })
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_idx = (selected_idx - 1) % len(MENU_OPTIONS)
                elif event.key == pygame.K_DOWN:
                    selected_idx = (selected_idx + 1) % len(MENU_OPTIONS)
                elif event.key == pygame.K_RETURN:
                    if selected_idx == 3:  # Custom
                        # Custom input
                        width, height = 8, 8
                        width_str, height_str = '', ''
                        active_field = 'width'
                        custom_done = False
                        while not custom_done:
                            draw_custom_input(screen, width_str, height_str, active_field)
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                elif event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_TAB:
                                        active_field = 'height' if active_field == 'width' else 'width'
                                    elif event.key == pygame.K_RETURN:
                                        if width_str.isdigit() and height_str.isdigit():
                                            width, height = int(width_str), int(height_str)
                                            custom_done = True
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
                        screen = run_comparison(screen, width, height)
                    elif selected_idx == 4:  # Quit
                        running = False
                    else:
                        width, height = DIFFICULTY_MAP[selected_idx]
                        screen = run_comparison(screen, width, height)
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(event.pos):
                        selected_idx = i
                        if i == 3:  # Custom
                            # Custom input (same as above)
                            width, height = 8, 8
                            width_str, height_str = '', ''
                            active_field = 'width'
                            custom_done = False
                            while not custom_done:
                                draw_custom_input(screen, width_str, height_str, active_field)
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        pygame.quit()
                                        sys.exit()
                                    elif event.type == pygame.KEYDOWN:
                                        if event.key == pygame.K_TAB:
                                            active_field = 'height' if active_field == 'width' else 'width'
                                        elif event.key == pygame.K_RETURN:
                                            if width_str.isdigit() and height_str.isdigit():
                                                width, height = int(width_str), int(height_str)
                                                custom_done = True
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
                            screen = run_comparison(screen, width, height)
                        elif i == 4:  # Quit
                            running = False
                        else:
                            width, height = DIFFICULTY_MAP[i]
                            screen = run_comparison(screen, width, height)
        clock.tick(30)
    pygame.quit()

if __name__ == '__main__':
    main() 