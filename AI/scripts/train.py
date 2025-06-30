import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pygame
from environment import MazeEnvironment
from scripts.dqn_model import DQNAgent
import torch
from GAME.configs.config import *

SCORES_DIR = os.path.join(os.path.dirname(__file__), 'AI Scores')
os.makedirs(SCORES_DIR, exist_ok=True)

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

def format_time(seconds):
    m = seconds // 60
    s = seconds % 60
    return f"{m}m{s}s" if m else f"{s}s"

def pad_state(state, target_size):
    if len(state) < target_size:
        return np.concatenate([state, np.zeros(target_size - len(state), dtype=state.dtype)])
    return state

def ai_play():
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
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model for {max_size}x{max_size}")
    for size in range(min_size, max_size+1):
        print(f"Level: {size}x{size}")
        run = 0
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
            # Dynamic time/step limits based on high score
            hs_time, hs_steps = load_highscore(size, size)
            if hs_time is not None and hs_steps is not None:
                time_limit = 10 * hs_time
                step_limit = 10 * hs_steps
            else:
                time_limit = 300
                step_limit = 500
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_r:
                            font = pygame.font.SysFont(None, 48)
                            msg = font.render("Time up! (Manual)", True, (200,0,0))
                            screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                            pygame.display.flip()
                            pygame.time.delay(1000)
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_p:
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
                            agent = DQNAgent(global_state_size, local_state_size, 4, device)
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_t:
                            show_trail = not show_trail
                            env.show_trail = show_trail
                if timed_out:
                    break
                padded_state = pad_state(state, global_state_size)
                action = agent.act(padded_state, local, training=True)
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
                reward_surf = font.render(f"Total Reward: {total_reward}", True, (255, 200, 0))
                screen.blit(reward_surf, (10, y))
                y += reward_surf.get_height() + 5
                # Show controls
                controls_font = pygame.font.SysFont(None, 24)
                controls = [
                    "Q: Quit",
                    "R: Time up (restart run)",
                    "P: Hard reset (clear scores, fresh agent)",
                    "T: Toggle trail"
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
                    reward_str = f"+{last_reward}"
                elif last_reward < 0:
                    reward_color = (200, 0, 0)
                    reward_str = f"{last_reward}"
                else:
                    reward_color = (180, 180, 180)
                    reward_str = f"0"
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
                if steps > step_limit or elapsed > time_limit:
                    font = pygame.font.SysFont(None, 48)
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
    print("All levels complete!")
    pygame.quit()

if __name__ == "__main__":
    ai_play() 