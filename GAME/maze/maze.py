import random

class Maze:
    def __init__(self, height, width, cell_size):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.maze = [['#'] * (2 * width + 1) for _ in range(2 * height + 1)]
        self._generate_maze()
        self.start = (1, 1)
        self.end = (2 * height - 1, 2 * width - 1)
        self.player = list(self.start)
        self.pixel_w = len(self.maze[0]) * self.cell_size
        self.pixel_h = len(self.maze) * self.cell_size

    def _generate_maze(self):
        visited = [[False] * self.width for _ in range(self.height)]
        stack = [(0, 0)]
        while stack:
            y, x = stack[-1]
            visited[y][x] = True
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(dirs)
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and not visited[ny][nx]:
                    self.maze[y * 2 + 1 + dy][x * 2 + 1 + dx] = ' '
                    self.maze[ny * 2 + 1][nx * 2 + 1] = ' '
                    stack.append((ny, nx))
                    break
            else:
                stack.pop()
        self.maze[1][1] = 'S'
        self.maze[2 * self.height - 1][2 * self.width - 1] = 'E'

    def move_player(self, direction):
        DIRS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        dy, dx = DIRS[direction]
        ny, nx = self.player[0] + dy, self.player[1] + dx
        if 0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0]) and self.maze[ny][nx] != '#':
            self.player = [ny, nx]
            return True
        return False

    def is_finished(self):
        return tuple(self.player) == self.end 