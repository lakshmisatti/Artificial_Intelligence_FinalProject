import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, game_name="Pong"):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((*state_size, action_size))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99998
        self.epsilon_min = 0.05
        self.model_path = f"{game_name}_qlearn.pkl"

    def get_action(self, state, exploit=False):
        if not exploit and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.q_table = pickle.load(f)


class ApproximateQLearningAgent:
    def __init__(self, num_actions, feature_count=7, game_name="Pong"):
        self.num_actions = num_actions
        self.weights = np.random.randn(feature_count, num_actions) * 0.01
        self.alpha = 0.05
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99998
        self.epsilon_min = 0.05
        self.model_path = f"{game_name}_approx.pkl"

    def normalize(self, features):
        norm = np.linalg.norm(features) + 1e-8
        return features / norm

    def get_action(self, features, exploit=False):
        features = self.normalize(features)
        q_values = np.dot(features, self.weights)
        if not exploit and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(q_values)

    def update(self, features, action, reward, next_features):
        features = self.normalize(features)
        next_features = self.normalize(next_features)
        q_current = np.dot(features, self.weights[:, action])
        q_next = np.max(np.dot(next_features, self.weights))
        target = reward + self.gamma * q_next
        td_error = target - q_current
        self.weights[:, action] += self.alpha * td_error * features
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.weights, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.weights = pickle.load(f)
