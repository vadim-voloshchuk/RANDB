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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()  # Loss function

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
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
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            
            # Обучаем модель
            self.optimizer.zero_grad()  # Обнуляем градиенты
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state).unsqueeze(0)))
            loss.backward()  # Обратное распространение
            self.optimizer.step()  # Шаг оптимизации

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

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
    
    # Размер состояния - 6 (положение и угол агента, положение и угол цели)
    agent = DQNAgent(state_size=6, action_size=5)

    episodes = 1000
    win_count = 0
    loss_count = 0

    for e in range(episodes):
        state, _ = env.reset()  # сброс среды
        # Объединение состояния агента и цели
        state = np.concatenate((
            state['agent'],           # 2D координаты агента (например, [x, y])
            state['target'],          # 2D координаты цели (например, [x, y])
            state['agent_angle'],     # Угол агента (одномерный массив)
            state['target_angle']     # Угол цели (одномерный массив)
        ))
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step({'move': action, 'view_angle': agent.epsilon})  # Использовать случайный угол
            # Объединение состояния для следующего шага
            next_state = np.concatenate((
                next_state['agent'],         # 2D координаты агента
                next_state['target'],        # 2D координаты цели
                next_state['agent_angle'],   # Угол агента
                next_state['target_angle']   # Угол цели
            ))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.memory) > 32:
                agent.replay(32)

        # Обновляем статистику выигрышей и проигрышей
        if reward == 100:  # Агент поймал цель
            win_count += 1
        elif reward == -100:  # Агент проиграл
            loss_count += 1

        print(f"Episode {e+1}/{episodes} finished. Reward: {episode_reward}. Wins: {win_count}, Losses: {loss_count}")

    agent.save("dqn_model.pth")  # Сохранить модель после обучения
