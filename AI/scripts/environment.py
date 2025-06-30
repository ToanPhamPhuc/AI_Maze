#region: Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pygame
from GAME.maze.maze import Maze
from GAME.configs.config import *
from collections import defaultdict
#endregion

#region: MazeEnvironment
class MazeEnvironment:
    """
    RL environment for the maze game, matching main game rendering and config logic.
    Returns both global state (full maze + player pos) and local wall state (up, down, left, right).
    Tracks and renders player trail as a heatmap.
    """
    #region: __init__
    def __init__(self, height=8, width=8, cell_size=None):
        self.height = height
        self.width = width
        self.cell_size = cell_size or DEFAULT_CELL_SIZE
        self.maze = None
        self.steps = 0
        self.trail = defaultdict(int)
        self.show_trail = True
        self.reset()
        self.state_size = len(self.maze.maze) * len(self.maze.maze[0]) + 2
        self.local_state_size = 4  # up, down, left, right
        self.action_space = 4
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    #endregion

    #region: reset
    def reset(self):
        self.maze = Maze(self.height, self.width, self.cell_size)
        self.steps = 0
        self.trail = defaultdict(int)
        self.trail[tuple(self.maze.player)] += 1
        self.last_pos = tuple(self.maze.player)
        self.min_dist_to_goal = self._manhattan_to_goal(self.maze.player)  # Track best distance this episode
        self.steps_since_progress = 0  # For stuck detection
        return self._get_state()
    #endregion

    #region: _get_state
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
    #endregion

    #region: get_local_wall_state
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
    #endregion

    #region: step
    def step(self, action):
        action_name = self.action_names[action]
        old_pos = self.maze.player.copy()
        old_dist = self._manhattan_to_goal(old_pos)
        moved = self.maze.move_player(action_name)
        if moved:
            self.steps += 1
            self.trail[tuple(self.maze.player)] += 1
        new_dist = self._manhattan_to_goal(self.maze.player)
        # New minimum distance bonus
        new_min_dist_bonus = 0
        stuck = False
        if new_dist < getattr(self, 'min_dist_to_goal', float('inf')):
            new_min_dist_bonus = 2.0
            self.min_dist_to_goal = new_dist
            self.steps_since_progress = 0  # Reset on progress
        else:
            self.steps_since_progress += 1
            if self.steps_since_progress > 100:
                stuck = True
        reward = self._calculate_reward(old_pos, moved, old_dist, new_dist) + new_min_dist_bonus
        done = self.maze.is_finished()
        self.last_pos = tuple(self.maze.player)
        next_state = self._get_state()
        info = {
            'moved': moved,
            'position': self.maze.player.copy(),
            'steps': self.steps,
            'stuck': stuck
        }
        return next_state, reward, done, info
    #endregion

    #region: _manhattan_to_goal
    def _manhattan_to_goal(self, pos):
        # Goal is at self.maze.end
        return abs(pos[0] - self.maze.end[0]) + abs(pos[1] - self.maze.end[1])
    #endregion

    #region: _calculate_reward
    def _calculate_reward(self, old_pos, moved, old_dist, new_dist):
        if self.maze.is_finished():
            return 100
        if not moved:
            return -10
        # Reward shaping: positive for getting closer, negative for getting further
        dist_reward = 0
        if new_dist < old_dist:
            dist_reward = 1.0  # reward for getting closer
        elif new_dist > old_dist:
            dist_reward = -1.0  # penalty for getting further
        # Mild revisit penalty
        revisit_penalty = -0.05 * (self.trail[tuple(self.maze.player)] - 1) if self.trail[tuple(self.maze.player)] > 1 else 0
        # Small step penalty
        step_penalty = -0.1
        # Exploration bonus for first visit
        exploration_bonus = 0.2 if self.trail[tuple(self.maze.player)] == 1 else 0
        return dist_reward + revisit_penalty + step_penalty + exploration_bonus
    #endregion

    #region: render
    def render(self, screen=None):
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
        # Draw trail as heatmap
        if self.show_trail and self.trail:
            base_alpha = 40
            step = 40
            max_alpha = 200
            for (ty, tx), count in self.trail.items():
                alpha = min(base_alpha + count * step, max_alpha)
                trail_color = (*PLAYER_TRAIL_COLOR[:3], alpha)
                trail_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                trail_surf.fill(trail_color)
                screen.blit(trail_surf, (offset_x + tx * self.cell_size, offset_y + ty * self.cell_size))
        # Draw player on top
        py, px = self.maze.player
        prect = pygame.Rect(offset_x + px * self.cell_size, offset_y + py * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, PLAYER_COLOR, prect)
        pygame.display.flip()
        return screen
    #endregion
#endregion 