import pygame
from GAME.configs.config import BG_COLOR, MENU_OPTIONS

def format_time(seconds):
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0 or d > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0 or d > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return ''.join(parts)

def draw_menu(screen, selected_idx, hover_idx=None, highscores=None):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    title = font.render("Select Difficulty", True, (255,255,0))
    options = MENU_OPTIONS
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
    option_rects = []
    for i, opt in enumerate(options):
        color = (255,255,255) if i == selected_idx or (hover_idx is not None and i == hover_idx) else (180,180,180)
        surf = small_font.render(opt, True, color)
        rect = surf.get_rect(center=(screen.get_width()//2, 220 + i*60))
        screen.blit(surf, rect)
        option_rects.append(rect)
        # Draw high score for this diff
        if highscores and i < 4:
            hs = highscores.get(opt.split()[0], None)
            if hs and hs[0] is not None:
                hs_surf = small_font.render(f"Best: {format_time(hs[0])}, {hs[1]} steps", True, (0,255,0))
                screen.blit(hs_surf, (screen.get_width()//2 + 180, 220 + i*60 - 16))
    pygame.display.flip()
    return option_rects 