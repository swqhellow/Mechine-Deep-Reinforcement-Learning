import numpy as np
import cv2
import random

# 设置窗口大小
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# 定义方块大小
BLOCK_SIZE = 2

# 定义方向
UP = (0, -BLOCK_SIZE)
DOWN = (0, BLOCK_SIZE)
LEFT = (-BLOCK_SIZE, 0)
RIGHT = (BLOCK_SIZE, 0)

class SnakeGame:
    def __init__(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False

    def _generate_food(self):
        while True:
            x = random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def step(self):
        if self.game_over:
            return

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # 判断是否撞墙
        if not (0 <= new_head[0] < WINDOW_WIDTH and 0 <= new_head[1] < WINDOW_HEIGHT):
            self.game_over = True
            return

        # 判断是否撞到自己
        if new_head in self.snake:
            self.game_over = True
            return

        # 移动蛇
        self.snake = [new_head] + self.snake[:-1]

        # 判断是否吃到食物
        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.food = self._generate_food()
            self.score += 1

    def change_direction(self, new_direction):
        # 防止蛇反向移动
        if (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
            self.direction = new_direction

    def draw(self, img):
        img[:] = BLACK

        # 画蛇
        for block in self.snake:
            cv2.rectangle(img, block, (block[0] + BLOCK_SIZE, block[1] + BLOCK_SIZE), GREEN, -1)

        # 画食物
        cv2.rectangle(img, self.food, (self.food[0] + BLOCK_SIZE, self.food[1] + BLOCK_SIZE), RED, -1)

        # 显示分数
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Score: {self.score}', (10, 30), font, 1, WHITE, 2, cv2.LINE_AA)

        if self.game_over:
            cv2.putText(img, 'Game Over', (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2), font, 2, RED, 3, cv2.LINE_AA)

def main():
    game = SnakeGame()
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.namedWindow('Snake')

    while True:
        game.step()
        game.draw(img)
        cv2.imshow('Snake', img)

        key = cv2.waitKey(100)
        if key == 27:  # ESC键
            break
        elif key == ord('w'):
            game.change_direction(UP)
        elif key == ord('s'):
            game.change_direction(DOWN)
        elif key == ord('a'):
            game.change_direction(LEFT)
        elif key == ord('d'):
            game.change_direction(RIGHT)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
