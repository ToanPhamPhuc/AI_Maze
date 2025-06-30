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
        
        # Action space: UP, DOWN, LEFT, RIGHT
        self.action_space = 4
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # State space: maze layout + player position
        self.state_size = self.height * self.width + 2  # maze cells + player x,y
        
    def reset(self):
        """Reset the environment to initial state."""
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.steps = 0  # Reset step count
        return self._get_state()
    
    def _get_state(self):
        """Convert current maze state to a numerical representation."""
        # Create a flattened representation of the maze
        state = []
        
        # Add maze layout (0 for path, 1 for wall)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.maze[y][x] == 1:  # Wall
                    state.append(1)
                else:  # Path
                    state.append(0)
        
        # Add player position (normalized)
        state.append(self.maze.player[0] / self.width)   # x position
        state.append(self.maze.player[1] / self.height)  # y position
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action and return (next_state, reward, done, info).
        
        Args:
            action: Integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        action_name = self.action_names[action]
        
        # Get current position
        old_pos = self.maze.player.copy()
        
        # Try to move
        moved = self.maze.move_player(action_name)
        self.steps += 1  # Increment step count
        
        # Calculate reward
        reward = self._calculate_reward(old_pos, moved)
        
        # Check if done
        done = self.maze.is_finished()
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'moved': moved,
            'position': self.maze.player.copy(),
            'steps': self.steps
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, old_pos, moved):
        """Calculate reward based on the action taken."""
        if self.maze.is_finished():
            return 100  # Big reward for reaching goal
        
        if not moved:
            return -10  # Penalty for hitting wall
        
        # Small penalty for each step to encourage efficiency
        return -1
    
    def render(self, screen=None):
        """Render the current state (for visualization)."""
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        
        screen.fill((255, 255, 255))  # White background
        
        # Draw maze
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.maze[y][x] == 1:  # Wall
                    pygame.draw.rect(screen, (0, 0, 0), 
                                   (x * self.cell_size, y * self.cell_size, 
                                    self.cell_size, self.cell_size))
        
        # Draw player
        pygame.draw.circle(screen, (255, 255, 0), 
                         (self.maze.player[0] * self.cell_size + self.cell_size // 2,
                          self.maze.player[1] * self.cell_size + self.cell_size // 2), 
                         self.cell_size // 3)
        
        # Draw goal
        pygame.draw.circle(screen, (255, 0, 0), 
                         (self.maze.end[0] * self.cell_size + self.cell_size // 2,
                          self.maze.end[1] * self.cell_size + self.cell_size // 2), 
                         self.cell_size // 3)
        
        pygame.display.flip()
        return screen 