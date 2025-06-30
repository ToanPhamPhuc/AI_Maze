import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GAME')))
import numpy as np
import pygame
from GAME.maze.maze import Maze
from GAME.configs.config import *

class MazeEnvironment:
    """
    Wrapper for the maze game to make it compatible with reinforcement learning.
    """
    
    def __init__(self, height=8, width=8, cell_size=30):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.maze = None
        self.steps = 0  # Track steps taken in the environment
        self.reset()
        # State space: full maze grid + player position
        self.state_size = len(self.maze.maze) * len(self.maze.maze[0]) + 2
        # Action space: UP, DOWN, LEFT, RIGHT
        self.action_space = 4
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def reset(self):
        """Reset the environment to initial state."""
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.steps = 0  # Reset step count
        return self._get_state()

    def _get_state(self):
        state = []
        for y in range(len(self.maze.maze)):
            for x in range(len(self.maze.maze[0])):
                if self.maze.maze[y][x] == '#':
                    state.append(1)
                else:
                    state.append(0)
        # Add player position (normalized)
        state.append(self.maze.player[0] / len(self.maze.maze[0]))
        state.append(self.maze.player[1] / len(self.maze.maze))
        return np.array(state, dtype=np.float32)

    def step(self, action):
        action_name = self.action_names[action]
        old_pos = self.maze.player.copy()
        moved = self.maze.move_player(action_name)  # Uses Maze logic, can't go through walls
        self.steps += 1
        reward = self._calculate_reward(old_pos, moved)
        done = self.maze.is_finished()
        next_state = self._get_state()
        info = {
            'moved': moved,
            'position': self.maze.player.copy(),
            'steps': self.steps
        }
        return next_state, reward, done, info

    def _calculate_reward(self, old_pos, moved):
        if self.maze.is_finished():
            return 100  # Big reward for reaching goal
        if not moved:
            return -10  # Penalty for hitting wall
        return -1  # Small penalty for each step

    def render(self, screen=None):
        grid_h = len(self.maze.maze)
        grid_w = len(self.maze.maze[0])
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((grid_w * self.cell_size, grid_h * self.cell_size))
        screen.fill((255, 255, 255))  # White background
        for y in range(grid_h):
            for x in range(grid_w):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if [y, x] == self.maze.player:
                    pygame.draw.rect(screen, (255, 255, 0), rect)  # Player (yellow)
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