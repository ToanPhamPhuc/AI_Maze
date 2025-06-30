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
    Wrapper for the maze game to make it compatible with reinforcement learning.
    """
    
    def __init__(self, height=3, width=3, cell_size=30):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.maze = None
        self.steps = 0  # Track steps taken in the environment
        self.trail = defaultdict(int)  # Track player trail as counts
        self.reset()
        # State space: full maze grid + player position
        self.state_size = len(self.maze.maze) * len(self.maze.maze[0]) + 2
        # Action space: UP, DOWN, LEFT, RIGHT
        self.action_space = 4
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        # Penalty strengths (tunable)
        self.start_penalty = -20
        self.trail_penalty_scale = -2
        self.wall_penalty = -3
        self.wall_streak_penalty = -2  # Additional penalty per repeated wall hit
        self.first_visit_reward = 3
        self.closer_to_goal_reward = 1
        self.no_wall_bonus = 1
        self.no_wall_steps_required = 5
        # Wall hit streak
        self.wall_hit_streak = 0
        self.last_wall_pos = None
        self.no_wall_steps = 0

    def reset(self):
        """Reset the environment to initial state."""
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.steps = 0  # Reset step count
        self.trail = defaultdict(int)
        self.trail[tuple(self.maze.player)] += 1
        self.last_pos = tuple(self.maze.player)
        self.last_dist_to_goal = self._distance_to_goal(self.maze.player)
        self.wall_hit_streak = 0
        self.last_wall_pos = None
        self.no_wall_steps = 0
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
        moved = self.maze.move_player(action_name)
        if moved:
            self.steps += 1
            self.trail[tuple(self.maze.player)] += 1
        reward = self._calculate_reward(old_pos, moved, action)
        done = self.maze.is_finished()
        self.last_pos = tuple(self.maze.player)
        self.last_dist_to_goal = self._distance_to_goal(self.maze.player)
        next_state = self._get_state()
        info = {
            'moved': moved,
            'position': self.maze.player.copy(),
            'steps': self.steps
        }
        return next_state, reward, done, info

    def _distance_to_goal(self, pos):
        # Manhattan distance to goal
        gy, gx = self.maze.end
        py, px = pos
        return abs(gy - py) + abs(gx - px)

    def _calculate_reward(self, old_pos, moved, action):
        if self.maze.is_finished():
            return 100  # Big reward for reaching goal
        if not moved:
            if self.last_wall_pos == (tuple(old_pos), action):
                self.wall_hit_streak += 1
            else:
                self.wall_hit_streak = 1
                self.last_wall_pos = (tuple(old_pos), action)
            self.no_wall_steps = 0
            return self.wall_penalty + self.wall_streak_penalty * (self.wall_hit_streak - 1)
        else:
            self.wall_hit_streak = 0
            self.last_wall_pos = None
            self.no_wall_steps += 1
        # No-wall bonus
        bonus = 0
        if self.no_wall_steps > 0 and self.no_wall_steps % self.no_wall_steps_required == 0:
            bonus += self.no_wall_bonus
        # Start cell penalty
        y, x = self.maze.player
        if self.maze.maze[y][x] == 'S' and [y, x] != old_pos:
            return self.start_penalty + bonus
        # First visit reward
        visits = self.trail[(y, x)]
        if visits == 1:
            return self.first_visit_reward + bonus
        # Trail penalty (heatmap)
        if visits > 1:
            return self.trail_penalty_scale * (visits - 1) + bonus
        # Closer to goal reward
        prev_dist = self._distance_to_goal(old_pos)
        curr_dist = self._distance_to_goal([y, x])
        if curr_dist < prev_dist:
            return self.closer_to_goal_reward + bonus
        return -1 + bonus

    def render(self, screen=None):
        grid_h = len(self.maze.maze)
        grid_w = len(self.maze.maze[0])
        # Calculate cell size and screen size as in main.py
        cell_w = MAX_SCREEN_W // grid_w
        cell_h = MAX_SCREEN_H // grid_h
        cell_size = min(cell_w, cell_h, DEFAULT_CELL_SIZE)
        maze_pixel_w = grid_w * cell_size
        maze_pixel_h = grid_h * cell_size
        screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w))
        screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h))
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((screen_width, screen_height))
        screen.fill(BG_COLOR)  # Use config background color
        # Center the maze
        offset_x = (screen_width - maze_pixel_w) // 2
        offset_y = (screen_height - maze_pixel_h) // 2
        # Draw maze background
        for y in range(grid_h):
            for x in range(grid_w):
                rect = pygame.Rect(offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
                cell = self.maze.maze[y][x]
                if cell == '#':
                    pygame.draw.rect(screen, WALL_COLOR, rect)
                elif cell == 'S':
                    pygame.draw.rect(screen, START_COLOR, rect)
                elif cell == 'E':
                    pygame.draw.rect(screen, EXIT_COLOR, rect)
                else:
                    pygame.draw.rect(screen, PATH_COLOR, rect)
        # Draw trail (with alpha by count)
        base_alpha = 40
        step = 40
        max_alpha = 200
        for (ty, tx), count in self.trail.items():
            if self.maze.maze[ty][tx] not in ('#',):
                alpha = min(base_alpha + count * step, max_alpha)
                trail_color = (*PLAYER_TRAIL_COLOR[:3], alpha)
                trail_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                trail_surf.fill(trail_color)
                screen.blit(trail_surf, (offset_x + tx * cell_size, offset_y + ty * cell_size))
        # Draw player always on top
        py, px = self.maze.player
        prect = pygame.Rect(offset_x + px * cell_size, offset_y + py * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, PLAYER_COLOR, prect)
        pygame.display.flip()
        return screen 