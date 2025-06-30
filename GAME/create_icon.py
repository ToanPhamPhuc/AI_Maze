from PIL import Image, ImageDraw
import os

# Create a 256x256 icon
size = 256
icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw maze-like pattern
cell_size = size // 16
wall_color = (100, 100, 255)  # Blue walls
path_color = (255, 255, 255)  # White paths
player_color = (255, 255, 0)  # Yellow player
goal_color = (255, 100, 100)  # Red goal

# Draw background paths
for y in range(0, size, cell_size):
    for x in range(0, size, cell_size):
        # Create maze-like pattern
        if (x // cell_size + y // cell_size) % 2 == 0:
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=path_color)
        else:
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=wall_color)

# Draw player (yellow circle)
player_x = cell_size * 2
player_y = cell_size * 2
draw.ellipse([player_x, player_y, player_x + cell_size - 1, player_y + cell_size - 1], fill=player_color)

# Draw goal (red circle)
goal_x = size - cell_size * 3
goal_y = size - cell_size * 3
draw.ellipse([goal_x, goal_y, goal_x + cell_size - 1, goal_y + cell_size - 1], fill=goal_color)

# Save as ICO file
icon_path = os.path.join('dist', 'maze_icon.ico')
if not os.path.exists('dist'):
    os.makedirs('dist')
icon.save(icon_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])

print("Maze icon created: " + icon_path) 