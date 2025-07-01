#region: Imports
import sys
import os
import glob
import time
import numpy as np
import pygame
from environment import MazeEnvironment
from dqn_model import DQNAgent
import torch
from GAME.configs.config import *
import json
#endregion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
SCORES_DIR = os.path.join(os.path.dirname(__file__), 'AI Scores')
os.makedirs(SCORES_DIR, exist_ok=True)
best_runs_dir = os.path.join(os.path.dirname(__file__), 'best_runs')
os.makedirs(best_runs_dir, exist_ok=True)

#region: Scores
def get_highscore_path(w, h):
    # Cleaned up to avoid hidden characters or encoding issues
    return os.path.join(SCORES_DIR, f'HighScore{w}x{h}')

def load_highscore(w, h):
    path = get_highscore_path(w, h)
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        line = f.read().strip()
        if not line:
            return None, None
        t, s = line.split(',')
        return int(t), int(s)

def save_highscore(w, h, time_sec, steps):
    path = get_highscore_path(w, h)
    with open(path, 'w') as f:
        f.write(f'{time_sec},{steps}')
#endregion

def format_time(seconds):
    m = seconds // 60
    s = seconds % 60
    return f"{m}m{s}s" if m else f"{s}s"

def pad_state(state, target_size):
    if len(state) < target_size:
        return np.concatenate([state, np.zeros(target_size - len(state), dtype=state.dtype)])
    return state

def list_best_runs(min_size, max_size):
    runs = []
    for size in range(min_size, max_size+1):
        path = os.path.join(best_runs_dir, f'best_run_{size}x{size}.json')
        runs.append(os.path.exists(path))
    return runs

def replay_best_run(size):
    path = os.path.join(best_runs_dir, f'best_run_{size}x{size}.json')
    if not os.path.exists(path):
        print(f"No best run for {size}x{size}")
        return
    with open(path, 'r') as f:
        data = json.load(f)
    actions = data['actions']
    steps = data.get('steps', len(actions))
    total_time = data.get('time', 0)
    # Setup environment
    env = MazeEnvironment(height=size, width=size)
    (state, local), done = env.reset(), False
    screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, (2*size+1)*DEFAULT_CELL_SIZE))
    screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, (2*size+1)*DEFAULT_CELL_SIZE))
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Replay: {size}x{size}")
    clock = pygame.time.Clock()
    idx = 0
    while not done and idx < len(actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        action = actions[idx]
        (next_state, next_local), reward, done, info = env.step(action)
        state, local = next_state, next_local
        screen = env.render(screen)
        font = pygame.font.SysFont(None, 32)
        msg = font.render(f"Step {idx+1}/{steps}", True, (255,255,0))
        screen.blit(msg, (10, 10))
        pygame.display.flip()
        pygame.time.delay(60)
        idx += 1
    # Show finish message
    font = pygame.font.SysFont(None, 48)
    msg = font.render("Replay finished! Press ESC to return.", True, (0,200,0))
    screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
    pygame.display.flip()
    # Wait for ESC
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        pygame.time.delay(100)

def main_menu():
    min_size, max_size = 3, 16
    pygame.init()
    screen = pygame.display.set_mode((600, 800))
    pygame.display.set_caption("Maze AI Trainer Menu")
    font = pygame.font.SysFont(None, 36)
    big_font = pygame.font.SysFont(None, 48)
    selected = 0
    menu_items = [f"Replay {size}x{size}" for size in range(min_size, max_size+1)]
    menu_items += ["Start Training", "Quit"]
    item_rects = []
    while True:
        screen.fill((30,30,30))
        runs = list_best_runs(min_size, max_size)
        y = 60
        title = big_font.render("Maze AI Trainer", True, (255,255,0))
        screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 10))
        item_rects = []
        for i, item in enumerate(menu_items):
            color = (255,255,255)
            if i == selected:
                color = (0,255,255)
            if i < max_size-min_size+1:
                label = f"{item}  [{'Replay' if runs[i] else 'No Replay'}]"
            else:
                label = item
            surf = font.render(label, True, color)
            rect = surf.get_rect(topleft=(60, y))
            screen.blit(surf, rect.topleft)
            item_rects.append(rect)
            y += 48
        pygame.display.flip()
        mouse_pos = pygame.mouse.get_pos()
        mouse_hover = None
        for i, rect in enumerate(item_rects):
            if rect.collidepoint(mouse_pos):
                mouse_hover = i
                break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_items)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_items)
                elif event.key == pygame.K_RETURN:
                    if selected < max_size-min_size+1:
                        if runs[selected]:
                            replay_best_run(selected+min_size)
                    elif menu_items[selected] == "Start Training":
                        return  # Proceed to training
                    elif menu_items[selected] == "Quit":
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.MOUSEMOTION:
                if mouse_hover is not None:
                    selected = mouse_hover
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mouse_hover is not None:
                    selected = mouse_hover
                    if selected < max_size-min_size+1:
                        if runs[selected]:
                            replay_best_run(selected+min_size)
                    elif menu_items[selected] == "Start Training":
                        return
                    elif menu_items[selected] == "Quit":
                        pygame.quit()
                        sys.exit()
        pygame.time.delay(100)

#region: main AI training
def ai_play():
    #region: Game init
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_size, max_size = 3, 16  # Use config or curriculum
    runs_per_level = 50
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    # Use the largest state size for the agent
    env = MazeEnvironment(height=max_size, width=max_size)
    global_state_size = env.state_size
    local_state_size = env.local_state_size
    agent = DQNAgent(global_state_size, local_state_size, 4, device)
    model_path = os.path.join(model_dir, f'maze_dqn_{max_size}x{max_size}.pth')
    os.makedirs(best_runs_dir, exist_ok=True)
    #endregion
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model for {max_size}x{max_size}")
    size = min_size
    while size <= max_size:
        print(f"Level: {size}x{size}")
        run = 0
        go_lower = False
        skip = False
        return_to_menu = False
        while run < runs_per_level:
            # Calculate cell size and window size as in main game
            maze_pixel_w = (2 * size + 1)
            maze_pixel_h = (2 * size + 1)
            cell_w = MAX_SCREEN_W // maze_pixel_w
            cell_h = MAX_SCREEN_H // maze_pixel_h
            cell_size = min(cell_w, cell_h, DEFAULT_CELL_SIZE)
            screen_width = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w * cell_size))
            screen_height = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h * cell_size))
            env = MazeEnvironment(height=size, width=size, cell_size=cell_size)
            env.show_trail = True
            (state, local), done = env.reset(), False
            steps = 0
            start_time = time.time()
            screen = pygame.display.set_mode((screen_width, screen_height))
            timed_out = False
            total_reward = 0
            last_reward = 0
            show_trail = True
            freeze = False
            # Dynamic time/step limits based on high score
            hs_time, hs_steps = load_highscore(size, size)
            if hs_time is not None and hs_steps is not None:
                if hs_time == 0:
                    time_limit = 10
                else:
                    time_limit = 10 * hs_time
                step_limit = 10 * hs_steps
            else:
                time_limit = 300
                step_limit = 500
            actions_this_run = []
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_ESCAPE:
                            return_to_menu = True
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_r:
                            font = pygame.font.SysFont(None, 48)
                            msg = font.render("Time up! (Manual)", True, (200,0,0))
                            screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                            pygame.display.flip()
                            pygame.time.delay(1000)
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_p: # key to reset everything: agent, scores, ...
                            for f in glob.glob(os.path.join(SCORES_DIR, 'HighScore*')):
                                try:
                                    os.remove(f)
                                except Exception:
                                    pass
                            font = pygame.font.SysFont(None, 48)
                            msg = font.render("Hard reset! All scores cleared.", True, (200,0,0))
                            screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                            pygame.display.flip()
                            pygame.time.delay(1500)
                            agent = DQNAgent(global_state_size, local_state_size, 4, device) #reset agent yo
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_t:
                            show_trail = not show_trail
                            env.show_trail = show_trail
                        if event.key == pygame.K_RETURN:
                            freeze = not freeze
                        if event.key == pygame.K_o:
                            font = pygame.font.SysFont(None, 48)
                            msg = font.render("Skipped to next size!", True, (0,128,255))
                            screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                            pygame.display.flip()
                            pygame.time.delay(1000)
                            timed_out = True
                            done = True
                            skip = True
                            print(f"[INFO] Skipped {size}x{size} maze.")
                            break
                        if event.key == pygame.K_l:
                            if size > min_size:
                                font = pygame.font.SysFont(None, 48)
                                msg = font.render("Returned to previous size!", True, (0,128,255))
                                screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                                pygame.display.flip()
                                pygame.time.delay(1000)
                                timed_out = True
                                done = True
                                go_lower = True
                                print(f"[INFO] Returned to previous size from {size}x{size} maze.")
                                break
                if timed_out:
                    break
                if freeze:
                    font = pygame.font.SysFont(None, 48)
                    msg = font.render("Frozen (Press Enter)", True, (0,200,255))
                    screen.blit(msg, (10, screen.get_height()//2 - msg.get_height()//2))
                    pygame.display.flip()
                    pygame.time.delay(100)
                    continue
                padded_state = pad_state(state, global_state_size)
                action = agent.act(padded_state, local, training=True)
                actions_this_run.append(int(action))
                (next_state, next_local), reward, done, info = env.step(action)
                padded_next_state = pad_state(next_state, global_state_size)
                agent.remember(padded_state, local, action, reward, padded_next_state, next_local, done)
                agent.replay()
                state, local = next_state, next_local
                steps = env.steps
                total_reward += reward
                last_reward = reward
                elapsed = int(time.time() - start_time)
                screen = env.render(screen)
                # Draw overlays and stats
                font = pygame.font.SysFont(None, 28)
                y = 0
                res_str = f"Resolution: {screen.get_width()}x{screen.get_height()}"
                res_surf = font.render(res_str, True, (255,255,255))
                screen.blit(res_surf, (10, y))
                y += res_surf.get_height() + 5
                steps_surf = font.render(f"Steps: {steps}", True, (255,255,255))
                screen.blit(steps_surf, (10, y))
                y += steps_surf.get_height() + 5
                time_surf = font.render(f"Time: {format_time(elapsed)}", True, (255,255,255))
                screen.blit(time_surf, (10, y))
                y += time_surf.get_height() + 5
                diff_str = f'{size}x{size}'
                diff_surf = font.render(f"Current: {diff_str}", True, (255,255,0))
                screen.blit(diff_surf, (10, y))
                y += diff_surf.get_height() + 5
                # Show time/step limits
                lim_surf = font.render(f"Limit: {format_time(time_limit)}, {step_limit} steps", True, (255,128,0))
                screen.blit(lim_surf, (10, y))
                y += lim_surf.get_height() + 5
                hs_time, hs_steps = load_highscore(size, size)
                if hs_time is not None:
                    hs_surf = font.render(f"Best: {format_time(hs_time)}, {hs_steps} steps", True, (0,255,0))
                    screen.blit(hs_surf, (10, y))
                    y += hs_surf.get_height() + 5
                reward_surf = font.render(f"Total Reward: {total_reward:.1f}", True, (255, 200, 0))
                screen.blit(reward_surf, (10, y))
                y += reward_surf.get_height() + 5
                # Show controls
                controls_font = pygame.font.SysFont(None, 24)
                controls = [
                    "Q: Quit",
                    "R: Time up (restart run)",
                    "P: Hard reset (clear scores, fresh agent)",
                    "T: Toggle trail",
                    "Enter: Freeze/Unfreeze",
                    "O: Skip to next size",
                    "L: Go to previous size"
                ]
                for i, text in enumerate(controls):
                    surf = controls_font.render(text, True, (255,255,0))
                    screen.blit(surf, (10, y + i*22))
                # Show last reward above player
                py, px = env.maze.player
                cell_size = env.cell_size
                reward_font = pygame.font.SysFont(None, 28)
                if last_reward > 0:
                    reward_color = (0, 200, 0)
                    reward_str = f"+{last_reward:.1f}"
                elif last_reward < 0:
                    reward_color = (200, 0, 0)
                    reward_str = f"{last_reward:.1f}"
                else:
                    reward_color = (180, 180, 180)
                    reward_str = f"0.0"
                # Centered above player
                grid_h = len(env.maze.maze)
                grid_w = len(env.maze.maze[0])
                maze_pixel_w = grid_w * cell_size
                maze_pixel_h = grid_h * cell_size
                win_w = max(MIN_SCREEN_W, min(MAX_SCREEN_W, maze_pixel_w))
                win_h = max(MIN_SCREEN_H, min(MAX_SCREEN_H, maze_pixel_h))
                offset_x = (win_w - maze_pixel_w) // 2 if win_w > maze_pixel_w else 0
                offset_y = (win_h - maze_pixel_h) // 2 if win_h > maze_pixel_h else 0
                rx = offset_x + px * cell_size + cell_size // 2
                ry = offset_y + py * cell_size - 28
                reward_head = reward_font.render(reward_str, True, reward_color)
                screen.blit(reward_head, (rx - reward_head.get_width() // 2, ry))
                pygame.display.flip()
                pygame.time.delay(60)
                # Dynamic time/step punishment
                if steps > step_limit or elapsed > time_limit or info.get('stuck', False):
                    font = pygame.font.SysFont(None, 48)
                    if info.get('stuck', False):
                        msg = font.render("Stuck! (Auto reset)", True, (200,0,0))
                    else:
                        msg = font.render("Time up!", True, (200,0,0))
                    screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                    pygame.display.flip()
                    pygame.time.delay(2000)
                    timed_out = True
                    break
            elapsed = int(time.time() - start_time)
            if not timed_out:
                hs_time, hs_steps = load_highscore(size, size)
                new_hs = False
                if hs_time is None or elapsed < hs_time or (elapsed == hs_time and steps < hs_steps):
                    save_highscore(size, size, elapsed, steps)
                    new_hs = True
                    # Save best run actions
                    best_run_path = os.path.join(best_runs_dir, f'best_run_{size}x{size}.json')
                    with open(best_run_path, 'w') as f:
                        json.dump({
                            'actions': actions_this_run,
                            'steps': steps,
                            'time': elapsed
                        }, f)
                print(f"Level {size}x{size} Run {run+1}/{runs_per_level}: Steps={steps}, Time={format_time(elapsed)}" + (" NEW HIGH SCORE!" if new_hs else ""))
                font = pygame.font.SysFont(None, 48)
                msg = font.render("Goal reached!", True, (0,200,0))
                screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                pygame.display.flip()
                pygame.time.delay(1200)
                if new_hs:
                    agent.save(model_path)
                    print(f"Model saved to {model_path} (new high score!)")
                run += 1
            else:
                print(f"Level {size}x{size} Run {run+1}/{runs_per_level}: Time up! (Steps={steps}, Time={format_time(elapsed)})")
            if go_lower:
                break
            if skip:
                break
            if return_to_menu:
                break
        if return_to_menu:
            main_menu()
            return
        if go_lower and size > min_size:
            size -= 1
            continue
        size += 1
    print("All levels complete!")
    pygame.quit()

if __name__ == "__main__":
    while True:
        main_menu()
        ai_play() 
#endregion