import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import redandblue



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0    # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        model.compile(optimizer=optim.Adam(model.parameters(), lr=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.model(torch.FloatTensor(state).unsqueeze(0), target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Пример использования агента
if __name__ == "__main__":
    env = gym.make(
            'RedAndBlue-v0.1',
            render_mode=None,
            size=100,
            fps=10,
            obstacle_type='random',
            obstacle_percentage=0.05,
            target_behavior='circle'
        )
    agent = DQNAgent(state_size=4, action_size=5)  # состояние может содержать координаты агента, координаты цели, угол и т. д.

    episodes = 1000
    for e in range(episodes):
        state, _ = env.reset()  # сброс среды
        state = np.concatenate((state['agent'], state['target'], state['agent_angle'], state['target_angle']))
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step({'move': action, 'view_angle': agent.epsilon})  # использовать случайный угол
            next_state = np.concatenate((next_state['agent'], next_state['target'], next_state['agent_angle'], next_state['target_angle']))
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > 32:
                agent.replay(32)

        print(f"Episode {e+1}/{episodes} finished")

    agent.save("dqn_model.pth")  # сохранить модель после обучения
