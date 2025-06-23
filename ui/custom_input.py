import pygame
from configs.config import BG_COLOR

def draw_custom_input(screen, width_str, height_str, active_field):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    title = font.render("Custom Maze Size", True, (255,255,0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
    w_label = small_font.render("Width:", True, (255,255,255))
    h_label = small_font.render("Height:", True, (255,255,255))
    w_color = (0,255,0) if active_field == 'width' else (255,255,255)
    h_color = (0,255,0) if active_field == 'height' else (255,255,255)
    w_input = small_font.render(width_str or "_", True, w_color)
    h_input = small_font.render(height_str or "_", True, h_color)
    screen.blit(w_label, (screen.get_width()//2 - 120, 200))
    screen.blit(w_input, (screen.get_width()//2, 200))
    screen.blit(h_label, (screen.get_width()//2 - 120, 260))
    screen.blit(h_input, (screen.get_width()//2, 260))
    instr = small_font.render("Enter numbers, press Enter to confirm", True, (180,180,180))
    screen.blit(instr, (screen.get_width()//2 - instr.get_width()//2, 340))
    pygame.display.flip() 