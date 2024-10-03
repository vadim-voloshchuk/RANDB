import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import redandblue
import pandas as pd

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
            nn.Linear(self.state_size, 128),  # Увеличьте размер для лучшей выразительности
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size + 1)  # +1 для предсказания угла
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.random.uniform(-180, 180)  # случайный угол
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        action = torch.argmax(act_values[0, :-1]).item()  # действия без угла
        predicted_angle = act_values[0, -1].item()  # предсказание угла
        return action, predicted_angle

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
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
    
    # Определяем state_size и action_size
    obstacle_grid_size = 100 * 100  # Размер матрицы препятствий
    state_size = 7 + obstacle_grid_size  # Размер состояния
    agent = DQNAgent(state_size=state_size, action_size=5)

    episodes = 5000
    win_count = 0
    loss_count = 0

    # Создание списка для хранения истории
    history = []

    for e in range(episodes):
        state, info = env.reset()  # сброс среды
        
        # Объединение состояния агента и цели
        obstacle_grid = np.zeros((100, 100))  # Создаем матрицу 100x100
        for obstacle in info['obstacles']:  # Предполагаем, что obstacles - это список координат
            x, y = obstacle  # Предполагаем, что obstacle - это кортеж (x, y)
            if 0 <= x < 100 and 0 <= y < 100:  # Проверяем границы
                obstacle_grid[x, y] = 1  # Обозначаем наличие препятствия

        # Плоское представление матрицы
        obstacle_flattened = obstacle_grid.flatten()

        state = np.concatenate((
            state['agent'],           # 2D координаты агента (например, [x, y])
            state['target'],          # 2D координаты цели (например, [x, y])
            state['agent_angle'],     # Угол агента (одномерный массив)
            state['target_angle'],     # Угол цели (одномерный массив)
            [info['distance']],    # Расстояние до цели
            obstacle_flattened  # Плоское представление матрицы препятствий
        ))

        print(f"Initial state shape: {state.shape}")

        done = False
        episode_reward = 0

        while not done:
            action, predicted_angle = agent.act(state)  # Получаем действие и предсказанный угол
            next_state, reward, done, _, info = env.step({'move': action, 'view_angle': predicted_angle})  # Используем предсказанный угол
            
            # Объединение состояния для следующего шага
            obstacle_grid = np.zeros((100, 100))  # Создаем матрицу 100x100
            for obstacle in info['obstacles']:  # Предполагаем, что obstacles - это список координат
                x, y = obstacle  # Предполагаем, что obstacle - это кортеж (x, y)
                if 0 <= x < 100 and 0 <= y < 100:  # Проверяем границы
                    obstacle_grid[x, y] = 1  # Обозначаем наличие препятствия

            next_state = np.concatenate((
                next_state['agent'],         # 2D координаты агента
                next_state['target'],        # 2D координаты цели
                next_state['agent_angle'],   # Угол агента
                next_state['target_angle'],   # Угол цели
                [info['distance']],    # Расстояние до цели
                obstacle_grid.flatten()  # Плоское представление матрицы препятствий
            ))

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.memory) > 32:
                agent.replay(32)
            
            print(f"Reward: {episode_reward}. Wins: {win_count}, Losses: {loss_count}")

        # Обновляем статистику выигрышей и проигрышей
        if reward == 100:  # Агент поймал цель
            win_count += 1
        elif reward == -100:  # Агент проиграл
            loss_count += 1

        # Добавление текущего эпизода в историю
        history.append({
            "episode": e + 1,
            "reward": episode_reward,
            "wins": win_count,
            "losses": loss_count,
            "epsilon": agent.epsilon
        })

        print(f"Episode {e+1}/{episodes} finished. Reward: {episode_reward}. Wins: {win_count}, Losses: {loss_count}")

    # Сохранение истории в DataFrame
    df = pd.DataFrame(history)

    # Сохранение таблицы в CSV файл
    df.to_csv("training_history.csv", index=False)

    agent.save("dqn_model.pth")  # Сохранить модель после обучения
