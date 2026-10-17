import random
from collections import defaultdict

ACTIONS = {"^": (-1, 0), "v": (1, 0), "<": (0, -1), ">": (0, 1)}
REWARDS = {"G": 10, "X": -10}
STEP_REWARD = -1


class GridWorld:
    def __init__(self, rows):
        self.rows = rows
        self.start = next((r, c) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell == "S")

    def is_open(self, r, c):
        return 0 <= r < len(self.rows) and 0 <= c < len(self.rows[r]) and self.rows[r][c] != "#"

    def step(self, state, action):
        dr, dc = ACTIONS[action]
        r, c = state[0] + dr, state[1] + dc
        if not self.is_open(r, c):
            r, c = state
        cell = self.rows[r][c]
        return (r, c), REWARDS.get(cell, STEP_REWARD), cell in REWARDS


class QLearningAgent:
    def __init__(self, learning_rate=0.5, discount=0.9, exploration=0.2, seed=0):
        self.q = defaultdict(float)
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = exploration
        self.rng = random.Random(seed)

    def best_action(self, state):
        return max(ACTIONS, key=lambda action: self.q[state, action])

    def choose(self, state):
        if self.rng.random() < self.exploration:
            return self.rng.choice(list(ACTIONS))
        return self.best_action(state)

    def learn(self, state, action, reward, next_state, done):
        future = 0 if done else max(self.q[next_state, a] for a in ACTIONS)
        target = reward + self.discount * future
        self.q[state, action] += self.learning_rate * (target - self.q[state, action])

    def run_episode(self, world, policy, learn=False, max_steps=100):
        state, total, path = world.start, 0, [world.start]
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done = world.step(state, action)
            if learn:
                self.learn(state, action, reward, next_state, done)
            state, total = next_state, total + reward
            path.append(state)
            if done:
                break
        return total, path

    def train(self, world, episodes=500):
        return [self.run_episode(world, self.choose, learn=True)[0] for _ in range(episodes)]

    def greedy_path(self, world):
        return self.run_episode(world, self.best_action)[1]

    def policy_map(self, world):
        return "\n".join(
            "".join(self.best_action((r, c)) if cell in ".S" else cell for c, cell in enumerate(row))
            for r, row in enumerate(world.rows)
        )
