import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pygame
import time
import threading
from collections import deque
import heapq
from Dijkstra.scripts.environment import MazeEnvironment
from GAME.configs.config import MIN_SCREEN_W, MIN_SCREEN_H, DEFAULT_CELL_SIZE

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
                    env.expanded.add(npos)
                    if on_expand:
                        on_expand(npos)
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
                    env.expanded.add(npos)
                    if on_expand:
                        on_expand(npos)
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
                    env.expanded.add(npos)
                    if on_expand:
                        on_expand(npos)
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
                    env.expanded.add(npos)
                    if on_expand:
                        on_expand(npos)
    env.solution_path = []
    return []

class BotRunner:
    def __init__(self, name, solve_func, color):
        self.name = name
        self.solve_func = solve_func
        self.color = color
        self.env = None
        self.solution_path = []
        self.expanded = set()
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.expanded_count = 0

    def reset(self, width, height):
        self.env = MazeEnvironment(height, width, DEFAULT_CELL_SIZE)
        self.env.reset()
        self.solution_path = []
        self.expanded = set()
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.expanded_count = 0

    def run(self):
        self.start_time = time.time()
        self.solution_path = self.solve_func(self.env)
        self.end_time = time.time()
        self.steps = len(self.solution_path) - 1 if self.solution_path else 0
        self.expanded_count = len(self.env.expanded)
        self.finished = True

def format_time(seconds):
    if seconds is None:
        return "N/A"
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return '{}m {}s {}ms'.format(m, s, ms)

def render_bot(screen, bot, x_offset, y_offset, cell_size):
    """Render a single bot's maze and status"""
    if not bot.env:
        return
    
    # Draw maze
    maze = bot.env.maze.maze
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            rect = pygame.Rect(x_offset + x * cell_size, y_offset + y * cell_size, cell_size, cell_size)
            cell = maze[y][x]
            if cell == '#':
                pygame.draw.rect(screen, (50, 50, 50), rect)  # Wall
            elif cell == 'S':
                pygame.draw.rect(screen, (0, 255, 0), rect)   # Start
            elif cell == 'E':
                pygame.draw.rect(screen, (255, 0, 0), rect)   # End
            else:
                pygame.draw.rect(screen, (200, 200, 200), rect)  # Path
    
    # Draw expanded nodes
    for (ey, ex) in bot.env.expanded:
        rect = pygame.Rect(x_offset + ex * cell_size, y_offset + ey * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (100, 150, 255), rect)
    
    # Draw solution path
    if bot.solution_path:
        for (py, px) in bot.solution_path:
            rect = pygame.Rect(x_offset + px * cell_size, y_offset + py * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, bot.color, rect, 2)
    
    # Draw player
    py, px = bot.env.maze.player
    prect = pygame.Rect(x_offset + px * cell_size, y_offset + py * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (255, 255, 0), prect)
    
    # Draw border
    pygame.draw.rect(screen, bot.color, (x_offset, y_offset, len(maze[0]) * cell_size, len(maze) * cell_size), 3)

def render_status(screen, bot, x_offset, y_offset, cell_size):
    """Render bot status information"""
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    
    # Bot name
    name_text = font.render(bot.name, True, bot.color)
    screen.blit(name_text, (x_offset, y_offset - 30))
    
    if bot.finished:
        # Status information
        elapsed = bot.end_time - bot.start_time if bot.start_time and bot.end_time else None
        time_text = small_font.render(f'Time: {format_time(elapsed)}', True, (255, 255, 255))
        steps_text = small_font.render(f'Steps: {bot.steps}', True, (255, 255, 255))
        expanded_text = small_font.render(f'Expanded: {bot.expanded_count}', True, (255, 255, 255))
        
        screen.blit(time_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 5))
        screen.blit(steps_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 20))
        screen.blit(expanded_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 35))
    else:
        # Running status
        status_text = small_font.render('Running...', True, (255, 255, 0))
        screen.blit(status_text, (x_offset, y_offset + len(bot.env.maze.maze) * cell_size + 5))

def main():
    pygame.init()
    
    # Create bots
    bots = [
        BotRunner("Dijkstra", solve_dijkstra, (0, 255, 0)),      # Green
        BotRunner("A*", solve_astar, (255, 0, 255)),             # Magenta
        BotRunner("DFS", solve_dfs, (0, 255, 255)),              # Cyan
        BotRunner("BFS", solve_bfs, (255, 255, 0))               # Yellow
    ]
    
    # Calculate screen size for 2x2 grid with minimum 1366x768
    maze_size = 10  # 10x10 maze for better visibility
    cell_size = 20  # Larger cells for better visibility
    maze_pixel_size = maze_size * 2 + 1  # Account for wall cells
    total_maze_size = maze_pixel_size * cell_size
    
    # Calculate minimum required screen size
    min_screen_width = total_maze_size * 2 + 100  # 2 mazes wide + padding
    min_screen_height = total_maze_size * 2 + 150  # 2 mazes high + status area
    
    # Ensure minimum resolution of 1366x768
    screen_width = max(1366, min_screen_width)
    screen_height = max(768, min_screen_height)
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Pathfinding Algorithms Comparison')
    clock = pygame.time.Clock()
    
    # Initialize bots with same maze
    for bot in bots:
        bot.reset(maze_size, maze_size)
    
    # Start all bots in separate threads
    threads = []
    for bot in bots:
        thread = threading.Thread(target=bot.run)
        thread.start()
        threads.append(thread)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Restart
                    # Wait for current threads to finish
                    for thread in threads:
                        thread.join()
                    
                    # Reset and restart all bots
                    for bot in bots:
                        bot.reset(maze_size, maze_size)
                    
                    threads = []
                    for bot in bots:
                        thread = threading.Thread(target=bot.run)
                        thread.start()
                        threads.append(thread)
        
        # Clear screen
        screen.fill((30, 30, 30))
        
        # Calculate centered positions for the mazes
        start_x = (screen_width - total_maze_size * 2) // 2
        start_y = (screen_height - total_maze_size * 2) // 2
        
        # Render all bots
        positions = [
            (start_x, start_y),                                    # Top-left: Dijkstra
            (start_x + total_maze_size, start_y),                  # Top-right: A*
            (start_x, start_y + total_maze_size),                  # Bottom-left: DFS
            (start_x + total_maze_size, start_y + total_maze_size) # Bottom-right: BFS
        ]
        
        for i, bot in enumerate(bots):
            render_bot(screen, bot, positions[i][0], positions[i][1], cell_size)
            render_status(screen, bot, positions[i][0], positions[i][1], cell_size)
        
        # Draw title
        title_font = pygame.font.SysFont(None, 36)
        title_text = title_font.render('Pathfinding Algorithms Comparison', True, (255, 255, 255))
        screen.blit(title_text, (screen_width//2 - title_text.get_width()//2, 10))
        
        # Draw instructions
        instruction_font = pygame.font.SysFont(None, 20)
        instructions = [
            'Press R to restart with new maze',
            'Press Q or ESC to quit'
        ]
        for i, instruction in enumerate(instructions):
            inst_text = instruction_font.render(instruction, True, (150, 150, 150))
            screen.blit(inst_text, (10, screen_height - 40 + i * 15))
        
        pygame.display.flip()
        clock.tick(30)
    
    # Wait for threads to finish
    for thread in threads:
        thread.join()
    
    pygame.quit()

if __name__ == '__main__':
    main() 