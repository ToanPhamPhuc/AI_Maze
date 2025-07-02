import pygame
from GAME.configs.config import (
    WALL_COLOR, PATH_COLOR, PLAYER_COLOR, EXIT_COLOR, START_COLOR, BG_COLOR, PLAYER_TRAIL_COLOR
)

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
        base_alpha = 40
        step = 40
        max_alpha = 200
        for (ty, tx), count in trail.items():
            alpha = min(base_alpha + count * step, max_alpha)
            trail_color = (*PLAYER_TRAIL_COLOR[:3], alpha)
            trail_surf = pygame.Surface((maze.cell_size, maze.cell_size), pygame.SRCALPHA)
            trail_surf.fill(trail_color)
            screen.blit(trail_surf, (offset_x + tx * maze.cell_size, offset_y + ty * maze.cell_size))
    # Draw player
    py, px = maze.player
    prect = pygame.Rect(offset_x + px * maze.cell_size, offset_y + py * maze.cell_size, maze.cell_size, maze.cell_size)
    pygame.draw.rect(screen, PLAYER_COLOR, prect) 