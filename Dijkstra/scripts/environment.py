#region: Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pygame
from GAME.maze.maze import Maze
from GAME.configs.config import *
from collections import deque
#endregion

#region: MazeEnvironment
class MazeEnvironment:
    """
    Maze environment for Dijkstra-based solving and rendering.
    No AI or RL logic remains.
    """
    def __init__(self, height=8, width=8, cell_size=None):
        self.height = height
        self.width = width
        self.cell_size = cell_size or DEFAULT_CELL_SIZE
        self.maze = None
        self.solution_path = []
        self.expanded = set()  # For visualizing expanded nodes
        self.reset()

    def reset(self):
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.solution_path = []
        self.expanded = set()
        return self.maze.player.copy()

    def solve_with_dijkstra(self, on_expand=None, delay=0, screen=None):
        """
        Returns the shortest path from start to goal using Dijkstra's algorithm as a list of (y, x) positions.
        If on_expand is provided, it is called with each expanded cell (y, x).
        """
        start = tuple(self.maze.player)
        goal = tuple(self.maze.end)
        maze = self.maze.maze
        h, w = len(maze), len(maze[0])
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        while queue:
            (y, x), path = queue.popleft()
            if (y, x) == goal:
                self.solution_path = path
                return path
            for dy, dx in dirs:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != '#':
                    npos = (ny, nx)
                    if npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [npos]))
                        self.expanded.add(npos)
                        if on_expand:
                            on_expand(npos)
                        if delay > 0 and screen is not None:
                            self.render(screen, show_solution=False, highlight=npos)
                            pygame.time.delay(delay)
        self.solution_path = []
        return []

    def render(self, screen=None, show_solution=True, highlight=None):
        grid_h = len(self.maze.maze)
        grid_w = len(self.maze.maze[0])
        maze_pixel_w = grid_w * self.cell_size
        maze_pixel_h = grid_h * self.cell_size
        # Center the maze in the window
        win_w = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w))
        win_h = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h))
        offset_x = (win_w - maze_pixel_w) // 2 if win_w > maze_pixel_w else 0
        offset_y = (win_h - maze_pixel_h) // 2 if win_h > maze_pixel_h else 0
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((win_w, win_h))
        screen.fill(BG_COLOR)
        # Draw maze
        for y in range(grid_h):
            for x in range(grid_w):
                rect = pygame.Rect(offset_x + x * self.cell_size, offset_y + y * self.cell_size, self.cell_size, self.cell_size)
                cell = self.maze.maze[y][x]
                if cell == '#':
                    pygame.draw.rect(screen, WALL_COLOR, rect)
                elif cell == 'S':
                    pygame.draw.rect(screen, START_COLOR, rect)
                elif cell == 'E':
                    pygame.draw.rect(screen, EXIT_COLOR, rect)
                else:
                    pygame.draw.rect(screen, PATH_COLOR, rect)
        # Draw expanded nodes
        for (ey, ex) in getattr(self, 'expanded', []):
            rect = pygame.Rect(offset_x + ex * self.cell_size, offset_y + ey * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, (80, 180, 255), rect)  # Light blue for expanded
        # Draw highlight (current expansion)
        if highlight:
            hy, hx = highlight
            rect = pygame.Rect(offset_x + hx * self.cell_size, offset_y + hy * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, (255, 255, 0), rect)
        # Draw solution path
        if show_solution and self.solution_path:
            for (py, px) in self.solution_path:
                rect = pygame.Rect(offset_x + px * self.cell_size, offset_y + py * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, (0, 255, 0), rect, 2)  # Green outline for solution
        # Draw player on top
        py, px = self.maze.player
        prect = pygame.Rect(offset_x + px * self.cell_size, offset_y + py * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, PLAYER_COLOR, prect)
        pygame.display.flip()
        return screen
#endregion 