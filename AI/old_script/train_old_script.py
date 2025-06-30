import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pygame
from environment_old_script import MazeEnvironment
from dqn_model_old_script import DQNAgent
import torch

# Directory for high scores
SCORES_DIR = os.path.join(os.path.dirname(__file__), 'AI Scores')
os.makedirs(SCORES_DIR, exist_ok=True)

def get_highscore_path(w, h):
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
    # Pad the state vector with zeros to match the largest maze size
    if len(state) < target_size:
        return np.concatenate([state, np.zeros(target_size - len(state), dtype=state.dtype)])
    return state

def ai_play():
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_size, max_size = 3, 16
    runs_per_level = 5
    cell_size = 40
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    # Use the largest state size for the agent
    max_state_size = (2*max_size+1)*(2*max_size+1)+2
    agent = DQNAgent(max_state_size, 4, device)
    # Optionally load the best model for the largest size if it exists
    model_path = os.path.join(model_dir, f'maze_dqn_{max_size}x{max_size}.pth')
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model for {max_size}x{max_size}")

    for size in range(min_size, max_size+1):
        print(f"Level: {size}x{size}")
        for run in range(runs_per_level):
            env = MazeEnvironment(height=size, width=size, cell_size=cell_size)
            state = env.reset()
            done = False
            steps = 0
            start_time = time.time()
            screen = None
            timed_out = False
            freeze = False
            show_trail = True
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
                            agent = DQNAgent(max_state_size, 4, device)
                            timed_out = True
                            done = True
                            break
                        if event.key == pygame.K_t:
                            show_trail = not show_trail
                        if event.key == pygame.K_RETURN:
                            freeze = not freeze
                if timed_out:
                    break
                if freeze:
                    font = pygame.font.SysFont(None, 48)
                    msg = font.render("Frozen (Press Enter)", True, (0,200,255))
                    screen.blit(msg, (10, screen.get_height()//2 - msg.get_height()//2))
                    pygame.display.flip()
                    pygame.time.delay(100)
                    continue
                # Pad state for the largest input size
                padded_state = pad_state(state, max_state_size)
                action = agent.act(padded_state, training=True)  # Enable exploration
                next_state, reward, done, info = env.step(action)
                padded_next_state = pad_state(next_state, max_state_size)
                agent.remember(padded_state, action, reward, padded_next_state, done)
                agent.replay()  # In-place training
                state = next_state
                steps += 1
                elapsed = int(time.time() - start_time)
                # Render
                screen = env.render(screen)
                # Draw stats
                font = pygame.font.SysFont(None, 32)
                steps_surf = font.render(f"Steps: {steps}", True, (255,255,0))
                time_surf = font.render(f"Time: {format_time(elapsed)}", True, (255,255,0))
                screen.blit(steps_surf, (10, 10))
                screen.blit(time_surf, (10, 40))
                hs_time, hs_steps = load_highscore(size, size)
                if hs_time is not None:
                    hs_surf = font.render(f"Best: {format_time(hs_time)}, {hs_steps} steps", True, (0,128,0))
                    screen.blit(hs_surf, (10, 70))
                pygame.display.flip()
                pygame.time.delay(60)
                # Check for time/step limit
                time_limit = 10 * hs_time if hs_time else 300  # fallback to 5 min if no high score
                step_limit = 10 * hs_steps if hs_steps else 500  # fallback to 500 steps if no high score
                if steps > step_limit or elapsed > time_limit:
                    font = pygame.font.SysFont(None, 48)
                    msg = font.render("Time up!", True, (200,0,0))
                    screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                    pygame.display.flip()
                    pygame.time.delay(2000)
                    timed_out = True
                    break  # Restart the run at the same level
            # On finish
            elapsed = int(time.time() - start_time)
            if not timed_out:
                hs_time, hs_steps = load_highscore(size, size)
                new_hs = False
                if hs_time is None or elapsed < hs_time or (elapsed == hs_time and steps < hs_steps):
                    save_highscore(size, size, elapsed, steps)
                    new_hs = True
                print(f"Level {size}x{size} Run {run+1}/{runs_per_level}: Steps={steps}, Time={format_time(elapsed)}" + (" NEW HIGH SCORE!" if new_hs else ""))
                # Show finish message
                font = pygame.font.SysFont(None, 48)
                msg = font.render("Goal reached!", True, (0,200,0))
                screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2, screen.get_height()//2 - 40))
                pygame.display.flip()
                pygame.time.delay(1200)
                # Save model only if new high score
                if new_hs:
                    agent.save(model_path)
                    print(f"Model saved to {model_path} (new high score!)")
                run += 1  # Only increment run if goal was reached
            else:
                print(f"Level {size}x{size} Run {run+1}/{runs_per_level}: Time up! (Steps={steps}, Time={format_time(elapsed)})")
    print("All levels complete!")
    pygame.quit()

if __name__ == "__main__":
    ai_play() 