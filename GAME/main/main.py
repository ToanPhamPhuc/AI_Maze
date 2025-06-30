import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import pygame
from configs.config import *
from maze.maze import Maze
from utils.highscore import get_highscore_filename, load_highscore, save_highscore
from ui.menu import draw_menu
from ui.custom_input import draw_custom_input
from ui.game_screen import draw_maze
from collections import defaultdict

# Directions: (dy, dx)
DIRS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
KEY_TO_DIR = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
}

CELL_SIZE = DEFAULT_CELL_SIZE  # Default, will be recalculated

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

def reset_maze(h, w, cell_size):
    maze = Maze(h, w, cell_size)
    finished = False
    steps = 0
    start_time = pygame.time.get_ticks()
    solve_time = None
    trail = defaultdict(int)
    return maze, finished, steps, start_time, solve_time, trail

def format_time(seconds):
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return ''.join(parts)

def main():
    print('--- Maze Game (Pygame) ---')
    pygame.init()
    screen = pygame.display.set_mode((MIN_SCREEN_W, MIN_SCREEN_H))
    pygame.display.set_caption('Maze Game')
    clock = pygame.time.Clock()
    fullscreen = False
    maximized = False
    windowed_size = (MIN_SCREEN_W, MIN_SCREEN_H)
    # Menu state
    menu_state = 'menu'  # 'menu', 'custom', 'game'
    selected_idx = 0
    width_str = ''
    height_str = ''
    active_field = 'width'
    h, w = 8, 8  # default
    cell_size = DEFAULT_CELL_SIZE
    maze = None
    finished = False
    move_dir = None
    move_delay = MOVE_DELAY
    last_move_time = 0
    show_trail = True
    steps = 0
    start_time = 0
    solve_time = None
    trail = defaultdict(int)
    hover_idx = None
    current_diff = 'Beginner'
    current_highscore = None
    highscores = {
        'Beginner': load_highscore('Beginner'),
        'Intermediate': load_highscore('Intermediate'),
        'Expert': load_highscore('Expert'),
    }
    custom_hs = (None, None)
    while True:
        if menu_state == 'menu':
            # Update high scores for menu
            highscores['Beginner'] = load_highscore('Beginner')
            highscores['Intermediate'] = load_highscore('Intermediate')
            highscores['Expert'] = load_highscore('Expert')
            option_rects = draw_menu(screen, selected_idx, hover_idx, highscores)
            mouse_pos = pygame.mouse.get_pos()
            hover_idx = None
            for i, rect in enumerate(option_rects):
                if rect.collidepoint(mouse_pos):
                    hover_idx = i
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_idx = (selected_idx - 1) % 5
                    elif event.key == pygame.K_DOWN:
                        selected_idx = (selected_idx + 1) % 5
                    elif event.key == pygame.K_RETURN:
                        idx = selected_idx
                        if idx == 0:
                            h, w = 8, 8
                            current_diff = 'Beginner'
                            current_highscore = highscores['Beginner']
                            menu_state = 'game'
                        elif idx == 1:
                            h, w = 16, 16
                            current_diff = 'Intermediate'
                            current_highscore = highscores['Intermediate']
                            menu_state = 'game'
                        elif idx == 2:
                            h, w = 16, 30
                            current_diff = 'Expert'
                            current_highscore = highscores['Expert']
                            menu_state = 'game'
                        elif idx == 3:
                            menu_state = 'custom'
                            width_str = ''
                            height_str = ''
                            active_field = 'width'
                        elif idx == 4:
                            pygame.quit()
                            sys.exit()
                    if event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT):
                        if not fullscreen:
                            fullscreen = True
                            maximized = False
                            windowed_size = screen.get_size()
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            fullscreen = False
                            screen = pygame.display.set_mode(windowed_size)
                    elif event.key == pygame.K_F11:
                        if not maximized:
                            maximized = True
                            fullscreen = False
                            windowed_size = screen.get_size()
                            info = pygame.display.Info()
                            screen = pygame.display.set_mode((info.current_w, info.current_h))
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hover_idx is not None:
                        selected_idx = hover_idx
                        idx = selected_idx
                        if idx == 0:
                            h, w = 8, 8
                            current_diff = 'Beginner'
                            current_highscore = highscores['Beginner']
                            menu_state = 'game'
                        elif idx == 1:
                            h, w = 16, 16
                            current_diff = 'Intermediate'
                            current_highscore = highscores['Intermediate']
                            menu_state = 'game'
                        elif idx == 2:
                            h, w = 16, 30
                            current_diff = 'Expert'
                            current_highscore = highscores['Expert']
                            menu_state = 'game'
                        elif idx == 3:
                            menu_state = 'custom'
                            width_str = ''
                            height_str = ''
                            active_field = 'width'
                        elif idx == 4:
                            pygame.quit()
                            sys.exit()
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
                                current_diff = 'Custom'
                                custom_hs = load_highscore('Custom', w, h)
                                current_highscore = custom_hs
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
                    if event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT):
                        if not fullscreen:
                            fullscreen = True
                            maximized = False
                            windowed_size = screen.get_size()
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            fullscreen = False
                            screen = pygame.display.set_mode(windowed_size)
                    elif event.key == pygame.K_F11:
                        if not maximized:
                            maximized = True
                            fullscreen = False
                            windowed_size = screen.get_size()
                            info = pygame.display.Info()
                            screen = pygame.display.set_mode((info.current_w, info.current_h))
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
        elif menu_state == 'game':
            # Calculate cell size and screen size
            maze_pixel_w = (2 * w + 1)
            maze_pixel_h = (2 * h + 1)
            cell_w = MAX_SCREEN_W // maze_pixel_w
            cell_h = MAX_SCREEN_H // maze_pixel_h
            cell_size = min(cell_w, cell_h, DEFAULT_CELL_SIZE)
            screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w * cell_size))
            screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h * cell_size))
            screen = pygame.display.set_mode((screen_width, screen_height))
            if fullscreen:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            elif maximized:
                info = pygame.display.Info()
                screen = pygame.display.set_mode((info.current_w, info.current_h))
            maze, finished, steps, start_time, solve_time, trail = reset_maze(h, w, cell_size)
            menu_state = 'playing'
            move_dir = None
            last_move_time = 0
            show_trail = True
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
                    if event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT):
                        if not fullscreen:
                            fullscreen = True
                            maximized = False
                            windowed_size = screen.get_size()
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            fullscreen = False
                            screen = pygame.display.set_mode(windowed_size)
                    elif event.key == pygame.K_F11:
                        if not maximized:
                            maximized = True
                            fullscreen = False
                            windowed_size = screen.get_size()
                            info = pygame.display.Info()
                            screen = pygame.display.set_mode((info.current_w, info.current_h))
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
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
                        trail[tuple(maze.player)] += 1
            screen.fill(BG_COLOR)
            draw_maze(screen, maze, trail if show_trail else None)
            font = pygame.font.SysFont(None, 28)
            elapsed = (solve_time if solve_time is not None else now) - start_time
            elapsed_sec = elapsed // 1000
            # Draw screen t and stats stacked vertically
            y = 0
            res_str = f"Resolution: {screen.get_width()}x{screen.get_height()}"
            res_font = pygame.font.SysFont(None, 28)
            res_surf = res_font.render(res_str, True, (255,255,255))
            screen.blit(res_surf, (10, y))
            y += res_surf.get_height() + 5
            steps_surf = font.render(f"Steps: {steps}", True, (255,255,255))
            screen.blit(steps_surf, (10, y))
            y += steps_surf.get_height() + 5
            time_surf = font.render(f"Time: {format_time(elapsed_sec)}", True, (255,255,255))
            screen.blit(time_surf, (10, y))
            y += time_surf.get_height() + 5
            diff_str = current_diff if current_diff != 'Custom' else f'Custom {w}x{h}'
            diff_surf = font.render(f"Current: {diff_str}", True, (255,255,0))
            screen.blit(diff_surf, (10, y))
            y += diff_surf.get_height() + 5
            hs_val = current_highscore if current_diff != 'Custom' else custom_hs
            if hs_val and hs_val[0] is not None:
                hs_surf = font.render(f"Best: {format_time(hs_val[0])}", True, (0,255,0))
                screen.blit(hs_surf, (10, y))
                y += hs_surf.get_height() + 5
            hint_alpha = 80 if not show_trail else 200
            trail_hint_surf = font.render("Press T to show trail", True, (255,0,0))
            trail_hint_surf = trail_hint_surf.convert_alpha()
            trail_hint_surf.set_alpha(hint_alpha)
            screen.blit(trail_hint_surf, (10, y))
            if finished:
                overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 160))
                screen.blit(overlay, (0, 0))
                big_font = pygame.font.SysFont(None, 48)
                msg1 = big_font.render("Congratulations!", True, (255,255,0))
                msg2 = font.render(f"Solved in {steps} steps, {format_time(elapsed_sec)}", True, (255,255,255))
                msg3 = font.render("Press R for a new maze", True, (255,255,255))
                screen.blit(msg1, (screen.get_width()//2 - msg1.get_width()//2, screen.get_height()//2 - 60))
                screen.blit(msg2, (screen.get_width()//2 - msg2.get_width()//2, screen.get_height()//2))
                screen.blit(msg3, (screen.get_width()//2 - msg3.get_width()//2, screen.get_height()//2 + 40))
            pygame.display.flip()
            if not finished and maze.is_finished():
                finished = True
                solve_time = pygame.time.get_ticks() - start_time
                # Update high score if beaten
                if current_diff == 'Custom':
                    prev_time, prev_steps = load_highscore('Custom', w, h)
                    update = False
                    if prev_time is None or elapsed_sec < prev_time:
                        update = True
                    elif elapsed_sec == prev_time and (prev_steps is None or steps < prev_steps):
                        update = True
                    if update:
                        save_highscore('Custom', elapsed_sec, steps, w, h)
                        custom_hs = (elapsed_sec, steps)
                        current_highscore = (elapsed_sec, steps)
                else:
                    prev_time, prev_steps = load_highscore(current_diff)
                    update = False
                    if prev_time is None or elapsed_sec < prev_time:
                        update = True
                    elif elapsed_sec == prev_time and (prev_steps is None or steps < prev_steps):
                        update = True
                    if update:
                        save_highscore(current_diff, elapsed_sec, steps)
                        highscores[current_diff] = (elapsed_sec, steps)
                        current_highscore = (elapsed_sec, steps)
            clock.tick(60)

if __name__ == '__main__':
    main()
