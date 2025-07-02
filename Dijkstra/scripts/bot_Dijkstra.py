import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pygame
import time
from Dijkstra.scripts.environment import MazeEnvironment
from GAME.ui.menu import draw_menu
from GAME.ui.custom_input import draw_custom_input
from GAME.configs.config import MENU_OPTIONS, SCORES_DIR, MIN_SCREEN_W, MIN_SCREEN_H


# Map menu options to maze sizes
DIFFICULTY_MAP = {
    0: (3, 3),   # Beginner
    1: (8, 8),   # Intermediate
    2: (16, 16), # Expert
}

# High score file names
SCORE_FILES = {
    0: os.path.join(SCORES_DIR, 'BeginnerHighScore.txt'),
    1: os.path.join(SCORES_DIR, 'IntermediateHighScore.txt'),
    2: os.path.join(SCORES_DIR, 'ExpertHighScore.txt'),
}

def load_highscores():
    highscores = {}
    for idx, path in SCORE_FILES.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                line = f.read().strip()
                if line:
                    t, s = line.split(',')
                    highscores[idx] = (float(t), int(s))
                else:
                    highscores[idx] = (None, None)
        else:
            highscores[idx] = (None, None)
    return highscores

def save_highscore(idx, time_val, steps):
    path = SCORE_FILES[idx]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{},{}'.format(time_val, steps))

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return '{}m {}s {}ms'.format(m, s, ms)

def animate_dijkstra_search(env, screen, delay=10):
    def on_expand(pos):
        env.render(screen, show_solution=False, highlight=pos)
        pygame.time.delay(delay)
    env.solve_with_dijkstra(on_expand=on_expand, delay=0, screen=screen)

def animate_solution(env, screen, delay=60):
    for pos in env.solution_path[1:]:
        env.maze.player = list(pos)
        env.render(screen, show_solution=True)
        pygame.time.delay(delay)

def main():
    pygame.init()
    screen = pygame.display.set_mode((MIN_SCREEN_W, MIN_SCREEN_H))
    pygame.display.set_caption('Dijkstra Maze Bot')
    clock = pygame.time.Clock()
    running = True
    selected_idx = 0
    highscores = load_highscores()
    while running:
        option_rects = draw_menu(screen, selected_idx, highscores={
            'Beginner': highscores[0],
            'Intermediate': highscores[1],
            'Expert': highscores[2],
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
                        run_dijkstra_bot(screen, width, height, None)
                        highscores = load_highscores()
                    elif selected_idx == 4:  # Quit
                        running = False
                    else:
                        width, height = DIFFICULTY_MAP[selected_idx]
                        run_dijkstra_bot(screen, width, height, selected_idx)
                        highscores = load_highscores()
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
                            run_dijkstra_bot(screen, width, height, None)
                            highscores = load_highscores()
                        elif i == 4:  # Quit
                            running = False
                        else:
                            width, height = DIFFICULTY_MAP[i]
                            run_dijkstra_bot(screen, width, height, i)
                            highscores = load_highscores()
        clock.tick(30)
    pygame.quit()

def run_dijkstra_bot(screen, width, height, score_idx):
    runs = 10
    best_time = None
    best_steps = None
    for run in range(1, runs+1):
        env = MazeEnvironment(height, width)
        env.reset()
        # Animate Dijkstra search (scan/expand)
        animate_dijkstra_search(env, screen, delay=10)
        # Now animate the final path
        start_time = time.time()
        steps = len(env.solution_path) - 1
        animate_solution(env, screen, delay=60)
        elapsed = time.time() - start_time
        # Show result
        show_result(screen, run, runs, elapsed, steps, env)
        # Allow quit between runs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return
        if best_time is None or elapsed < best_time:
            best_time = elapsed
            best_steps = steps
        pygame.time.delay(800)
    # Save high score if better
    if score_idx is not None:
        prev_time, prev_steps = load_highscores()[score_idx]
        if prev_time is None or best_time < prev_time:
            save_highscore(score_idx, best_time, best_steps)

def show_result(screen, run, runs, elapsed, steps, env):
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    screen.fill((30,30,30))
    msg = font.render('Run {}/{}'.format(run, runs), True, (255,255,0))
    tmsg = small_font.render('Time: {}'.format(format_time(elapsed)), True, (255,255,255))
    smsg = small_font.render('Steps: {}'.format(steps), True, (255,255,255))
    dmsg = small_font.render('Distance: {}'.format(steps), True, (255,255,255))
    emsg = small_font.render('Expanded: {}'.format(len(env.expanded)), True, (255,255,255))
    screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, 200))
    screen.blit(tmsg, (screen.get_width()//2 - tmsg.get_width()//2, 300))
    screen.blit(smsg, (screen.get_width()//2 - smsg.get_width()//2, 350))
    screen.blit(dmsg, (screen.get_width()//2 - dmsg.get_width()//2, 400))
    screen.blit(emsg, (screen.get_width()//2 - emsg.get_width()//2, 450))
    pygame.display.flip()

if __name__ == '__main__':
    main() 