import numpy as np
import cv2
import random
from collections import deque

class PongGame:
    def __init__(self, width=400, height=400, grid_size=15, headless=False):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.headless = headless
        self.paddle_height = 50
        self.paddle_width = 10
        self.paddle_speed = 10
        self.ball_radius = 5
        self.actions = [-10, 0, 10]  
        self.ball_speed_x = 5
        self.ball_speed_y = 5
        self.max_speed = np.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)
        self.reset()

    def reset(self):
        self.agent_y = self.height // 2
        self.ai_y = self.height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = self.ball_speed_x * random.choice([-1, 1])
        self.ball_dy = self.ball_speed_y * random.choice([-1, 1])
        self.terminal = False

    def step(self, action):
        # Move agent paddle
        self.agent_y += self.actions[action]
        self.agent_y = np.clip(self.agent_y, 0, self.height - self.paddle_height)

        # Move AI paddle
        if self.ball_y > self.ai_y + self.paddle_height // 2:
            self.ai_y += self.paddle_speed * 0.9
        elif self.ball_y < self.ai_y + self.paddle_height // 2:
            self.ai_y -= self.paddle_speed * 0.9
        self.ai_y = np.clip(self.ai_y, 0, self.height - self.paddle_height)

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        reward = -0.001

        # Bounce off top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy *= -1

        # Left paddle (AI)
        if self.ball_x <= self.paddle_width:
            if self.ai_y <= self.ball_y <= self.ai_y + self.paddle_height:
                self.ball_dx *= -1
            else:
                self.terminal = True
                reward = 1  # agent scored

        # Right paddle (agent)
        if self.ball_x >= self.width - self.paddle_width:
            if self.agent_y <= self.ball_y <= self.agent_y + self.paddle_height:
                self.ball_dx *= -1
                reward = 5  # good hit
            else:
                self.terminal = True
                reward = -5  # missed

        return reward

    def render(self):
        if self.headless:
            return
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(self.ai_y)), (self.paddle_width, int(self.ai_y + self.paddle_height)), (255, 255, 255), -1)
        cv2.rectangle(frame, (self.width - self.paddle_width, self.agent_y), (self.width, self.agent_y + self.paddle_height), (255, 255, 255), -1)
        cv2.circle(frame, (self.ball_x, self.ball_y), self.ball_radius, (0, 255, 0), -1)
        cv2.imshow("Pong", frame)
        cv2.waitKey(1)

    def is_terminal(self):
        return self.terminal

    def get_num_actions(self):
        return len(self.actions)

    def get_state_space_size(self):
        return (self.grid_size, self.grid_size, self.grid_size, self.grid_size)

    def discretize_state(self):
        bx = min(self.grid_size - 1, self.ball_x * self.grid_size // self.width)
        by = min(self.grid_size - 1, self.ball_y * self.grid_size // self.height)
        py = min(self.grid_size - 1, self.agent_y * self.grid_size // self.height)
        dy = (self.ball_dy + 5) // 2  # crude direction encoding
        return (bx, by, py, dy)

    def get_state_features(self):
        features = []

        # Normalize dimensions (assuming width/height = 400)
        norm = lambda x, max_val: x / max_val

        # Positions
        ball_x = norm(self.ball_x, self.width)
        ball_y = norm(self.ball_y, self.height)
        paddle_y = norm(self.agent_y, self.height)

        # Velocities
        vx = self.ball_dx / self.max_speed
        vy = self.ball_dy / self.max_speed

        # Features
        f1 = norm(abs(self.ball_y - (self.agent_y + self.paddle_height / 2)), self.height)
        f2 = vx
        f3 = vy
        f4 = 1 if self.ball_dx < 0 else 0  # assuming agent paddle is on the left
        f5 = paddle_y
        f6 = ball_y
        f7 = 1.0  # bias term

        features = np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)
        return features

class PacmanGame:
    def __init__(self, headless=False):
        self.width = 5
        self.height = 5
        self.cell_size = 80
        self.actions = ['left', 'right', 'up', 'down']
        self.action_map = {0: (-1, 0),  # left
                          1: (1, 0),   # right
                          2: (0, -1),  # up
                          3: (0, 1)}   # down
        self.valid_actions = 4
        self.headless = headless
        
        # Exact wall configuration
        self.walls = {(0,0), (0,1), (0,3), (0,4),
                     (1,0), (1,4),
                     (2,2),
                     (3,0), (3,4),
                     (4,0), (4,1), (4,3), (4,4)}
        
        # Initialize game elements
        self.reset()
        
        # Visualization setup
        if not self.headless:
            cv2.namedWindow('Pac-Man', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pac-Man', self.width*self.cell_size, self.height*self.cell_size)

    def _create_food(self):
        food = set()
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.walls and (x, y) != (2, 4) and (x, y) != (2, 0):
                    food.add((x, y))
        return food

    def reset(self):
        """Reset the game state"""
        self.pacman = (2, 4)  # Starting position
        self.ghost = (2, 0)   # Ghost position
        self.food = self._create_food()
        self.score = 0
        self.direction = 'right'
        self.visited_cells = set()
        self.visited_cells.add((2, 4))
        self.steps_since_last_food = 0
        self.last_action = None
        self.consecutive_non_food = 0

    def get_num_actions(self):
        return self.valid_actions

    def discretize_state(self):
        """Simplified state representation for Q-learning"""
        px = min(1, self.pacman[0] // 3)  # 0 or 1
        py = min(1, self.pacman[1] // 3)  # 0 or 1
        food_near = 1 if any(abs(f[0]-self.pacman[0]) + abs(f[1]-self.pacman[1]) <= 1 for f in self.food) else 0
        return (px, py, food_near)

    def get_state_space_size(self):
        return (2, 2, 2)

    def get_state_features(self):
        features = np.zeros(14, dtype=np.float32)
        px, py = self.pacman
        gx, gy = self.ghost
        
        # 1-2: Normalized position
        features[0] = px / (self.width - 1)
        features[1] = py / (self.height - 1)
        
        # 3-6: Movement directions (strong bias)
        if self.direction == 'left':
            features[2] = 1.5
        elif self.direction == 'right':
            features[3] = 1.5
        elif self.direction == 'up':
            features[4] = 1.5
        elif self.direction == 'down':
            features[5] = 1.5
        
        # 7-10: Food information (strong food-seeking)
        if len(self.food) > 0:
            nearest_food = min(self.food, key=lambda f: abs(f[0]-px) + abs(f[1]-py))
            features[6] = (nearest_food[0] - px) / self.width  # Direction to food x
            features[7] = (nearest_food[1] - py) / self.height  # Direction to food y
            features[8] = 1.0 / (abs(nearest_food[0]-px) + abs(nearest_food[1]-py) + 1)  # Inverse distance
            features[9] = 1.0 if (abs(nearest_food[0]-px) + abs(nearest_food[1]-py)) < 2 else 0.0  # Food close
        
        # 11-14: Ghost and exploration
        features[10] = (gx - px) / self.width  # Ghost x
        features[11] = (gy - py) / self.height  # Ghost y
        features[12] = 1.0 if self.consecutive_non_food > 5 else 0.0  # Exploration push
        features[13] = len(self.food) / len(self._create_food())  # Food remaining
        
        return features

    def step(self, action):
        # Convert and validate action
        action = int(action) % 4
        self.last_action = action
        
        # Get movement vector
        dx, dy = self.action_map[action]
        
        # Calculate new position with wall collision check
        prev_pos = self.pacman
        new_x = (self.pacman[0] + dx) % self.width
        new_y = (self.pacman[1] + dy) % self.height
        
        if (new_x, new_y) not in self.walls:
            self.pacman = (new_x, new_y)
            self.direction = ['left', 'right', 'up', 'down'][action]
            self.visited_cells.add(self.pacman)
        
        # Move ghost (slower and less aggressive)
        self._move_ghost()
        
        # Calculate rewards with strong food-seeking incentives
        reward = -0.1  # Small time penalty
        
        # Food collection
        if self.pacman in self.food:
            self.food.remove(self.pacman)
            self.score += 10
            reward += 25  # Large food reward
            self.steps_since_last_food = 0
            self.consecutive_non_food = 0
        else:
            self.steps_since_last_food += 1
            self.consecutive_non_food += 1
        
        # Ghost interaction
        ghost_dist = abs(self.ghost[0]-self.pacman[0]) + abs(self.ghost[1]-self.pacman[1])
        if ghost_dist < 2:
            reward -= 3
        if self.pacman == self.ghost:
            reward -= 20
        
        # Reward moving toward food
        if len(self.food) > 0:
            nearest_food = min(self.food, key=lambda f: abs(f[0]-prev_pos[0]) + abs(f[1]-prev_pos[1]))
            old_dist = abs(nearest_food[0]-prev_pos[0]) + abs(nearest_food[1]-prev_pos[1])
            new_dist = abs(nearest_food[0]-self.pacman[0]) + abs(nearest_food[1]-self.pacman[1])
            if new_dist < old_dist:
                reward += 4.0  # Strong reward for moving toward food
            elif new_dist > old_dist:
                reward -= 2.0  # Penalty for moving away
        
        # Exploration bonus
        if (new_x, new_y) not in self.visited_cells:
            reward += 3.0
        
        # Force exploration if stuck
        if self.consecutive_non_food > 10:
            reward -= 2.0
            # Try to force movement toward unexplored areas
            unexplored = [a for a in range(4) if 
                         ((self.pacman[0] + self.action_map[a][0]) % self.width,
                          (self.pacman[1] + self.action_map[a][1]) % self.height) not in self.visited_cells
                         and ((self.pacman[0] + self.action_map[a][0]) % self.width,
                              (self.pacman[1] + self.action_map[a][1]) % self.height) not in self.walls]
            if unexplored and random.random() < 0.7:
                action = random.choice(unexplored)
                dx, dy = self.action_map[action]
                new_x = (self.pacman[0] + dx) % self.width
                new_y = (self.pacman[1] + dy) % self.height
                if (new_x, new_y) not in self.walls:
                    self.pacman = (new_x, new_y)
                    self.direction = ['left', 'right', 'up', 'down'][action]
        
        return reward

    def _move_ghost(self):
        """Ghost movement that won't block Pacman's path"""
        if random.random() < 0.3:  # 30% chance to move (slower ghost)
            x, y = self.ghost
            valid_moves = []
            for dx, dy in self.action_map.values():
                new_x, new_y = (x + dx) % self.width, (y + dy) % self.height
                if (new_x, new_y) not in self.walls:
                    valid_moves.append((new_x, new_y))
            
            if valid_moves:
                # 40% chance to chase, 60% random (less aggressive)
                if random.random() < 0.4:
                    target = self.pacman
                    chosen_move = min(valid_moves, 
                                    key=lambda m: abs(m[0]-target[0]) + abs(m[1]-target[1]))
                else:
                    chosen_move = random.choice(valid_moves)
                self.ghost = chosen_move

    def is_terminal(self):
        return self.pacman == self.ghost or not self.food

    def render(self):
        """Clear visualization showing movement and food"""
        img = np.zeros((self.height*self.cell_size, self.width*self.cell_size, 3), dtype=np.uint8)
        
        # Draw walls
        for x, y in self.walls:
            cv2.rectangle(img, 
                         (x*self.cell_size, y*self.cell_size),
                         ((x+1)*self.cell_size, (y+1)*self.cell_size),
                         (0, 128, 255), -1)
        
        # Draw food (larger and brighter for visibility)
        for x, y in self.food:
            cv2.circle(img, 
                      (x*self.cell_size + self.cell_size//2, 
                       y*self.cell_size + self.cell_size//2),
                      self.cell_size//5, (255, 255, 0), -1)
        
        # Draw Pacman with clear direction
        angle = {'left': 180, 'right': 0, 'up': 90, 'down': 270}.get(self.direction, 0)
        cv2.ellipse(img,
                  (self.pacman[0]*self.cell_size + self.cell_size//2,
                   self.pacman[1]*self.cell_size + self.cell_size//2),
                  (self.cell_size//3, self.cell_size//3), angle, 30, 330,
                  (0, 255, 255), -1)
        
        # Draw Ghost
        cv2.circle(img,
                  (self.ghost[0]*self.cell_size + self.cell_size//2,
                   self.ghost[1]*self.cell_size + self.cell_size//2),
                  self.cell_size//3, (0, 0, 255), -1)
        
        # Draw information
        cv2.putText(img, f"Score: {self.score}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Food left: {len(self.food)}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.last_action is not None:
            cv2.putText(img, f"Last move: {['left','right','up','down'][self.last_action]}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not self.headless:
            cv2.imshow('Pac-Man', img)
            cv2.waitKey(300)
        
        return img

    def __del__(self):
        if not self.headless:
            cv2.destroyAllWindows()
