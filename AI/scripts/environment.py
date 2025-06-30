import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GAME')))
import numpy as np
import pygame
from GAME.maze.maze import Maze
from GAME.configs.config import *
from collections import defaultdict

class MazeEnvironment:
    """
    New environment: returns both global state (full maze + player pos) and local wall state (up, down, left, right)
    """
    def __init__(self, height=8, width=8, cell_size=30):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.maze = None
        self.steps = 0
        self.trail = defaultdict(int)
        self.reset()
        self.state_size = len(self.maze.maze) * len(self.maze.maze[0]) + 2
        self.local_state_size = 4  # up, down, left, right
        self.action_space = 4
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def reset(self):
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.steps = 0
        self.trail = defaultdict(int)
        self.trail[tuple(self.maze.player)] += 1
        self.last_pos = tuple(self.maze.player)
        return self._get_state()

    def _get_state(self):
        # Global state: maze grid (1 for wall, 0 for path), player pos normalized
        state = []
        for y in range(len(self.maze.maze)):
            for x in range(len(self.maze.maze[0])):
                if self.maze.maze[y][x] == '#':
                    state.append(1)
                else:
                    state.append(0)
        state.append(self.maze.player[0] / len(self.maze.maze[0]))
        state.append(self.maze.player[1] / len(self.maze.maze))
        # Local wall state: up, down, left, right (1 if wall, 0 if open)
        local = self.get_local_wall_state()
        return (np.array(state, dtype=np.float32), np.array(local, dtype=np.float32))

    def get_local_wall_state(self):
        y, x = self.maze.player
        maze = self.maze.maze
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]  # UP, DOWN, LEFT, RIGHT
        state = []
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]):
                state.append(1 if maze[ny][nx] == '#' else 0)
            else:
                state.append(1)  # Out-of-bounds is wall
        return state

    def step(self, action):
        action_name = self.action_names[action]
        old_pos = self.maze.player.copy()
        moved = self.maze.move_player(action_name)
        if moved:
            self.steps += 1
            self.trail[tuple(self.maze.player)] += 1
        reward = self._calculate_reward(old_pos, moved)
        done = self.maze.is_finished()
        self.last_pos = tuple(self.maze.player)
        next_state = self._get_state()
        info = {
            'moved': moved,
            'position': self.maze.player.copy(),
            'steps': self.steps
        }
        return next_state, reward, done, info

    def _calculate_reward(self, old_pos, moved):
        if self.maze.is_finished():
            return 100
        if not moved:
            return -10
        return -1

    def render(self, screen=None):
        grid_h = len(self.maze.maze)
        grid_w = len(self.maze.maze[0])
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((grid_w * self.cell_size, grid_h * self.cell_size))
        screen.fill((255, 255, 255))
        for y in range(grid_h):
            for x in range(grid_w):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if [y, x] == self.maze.player:
                    pygame.draw.rect(screen, (255, 255, 0), rect)
                elif self.maze.maze[y][x] == '#':
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                elif self.maze.maze[y][x] == 'S':
                    pygame.draw.rect(screen, (255, 200, 0), rect)
                elif self.maze.maze[y][x] == 'E':
                    pygame.draw.rect(screen, (0, 200, 0), rect)
                else:
                    pygame.draw.rect(screen, (220, 220, 220), rect)
        pygame.display.flip()
        return screen 