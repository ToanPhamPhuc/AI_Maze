import os
from GAME.configs.config import SCORES_DIR

def get_highscore_filename(diff, w=None, h=None):
    base = SCORES_DIR
    if not os.path.exists(base):
        os.makedirs(base)
    if diff == 'Beginner':
        return os.path.join(base, 'BeginnerHighScore.txt')
    elif diff == 'Intermediate':
        return os.path.join(base, 'IntermediateHighScore.txt')
    elif diff == 'Expert':
        return os.path.join(base, 'ExpertHighScore.txt')
    elif diff == 'Custom' and w and h:
        return os.path.join(base, f'Custom{w}x{h}HighScore.txt')
    return None

def load_highscore(diff, w=None, h=None):
    fname = get_highscore_filename(diff, w, h)
    if fname and os.path.exists(fname):
        try:
            with open(fname, 'r') as f:
                line = f.read().strip()
                if ',' in line:
                    t, s = line.split(',')
                    return int(t), int(s)
                else:
                    return int(line), None
        except:
            return None, None
    return None, None

def save_highscore(diff, time_score, steps_score, w=None, h=None):
    fname = get_highscore_filename(diff, w, h)
    if fname:
        with open(fname, 'w') as f:
            f.write(f'{time_score},{steps_score}') 