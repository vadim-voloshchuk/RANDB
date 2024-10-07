import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import gymnasium as gym
import pygame
import redandblue
from torch.cuda.amp import autocast, GradScaler  # Для использования AMP
import threading
import matplotlib
import matplotlib.animation as animation
import pandas as pd  # Импортируем pandas для работы с таблицей


matplotlib.use('TkAgg')

# Использование GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функция потерь Huber
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss).mean()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, angle_dim=1, grid_size=100):
        super(DQN, self).__init__()
        
        # CNN для обработки карты препятствий
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Полносвязные слои для остальной информации о состоянии
        self.grid_size = grid_size
        cnn_output_dim = self.calculate_cnn_output_size(grid_size)  # Вычисляем размер
        combined_dim = cnn_output_dim + input_dim  # Исправление: убираем вычитание
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, output_dim)
        self.angle_head = nn.Linear(128, angle_dim)

    def calculate_cnn_output_size(self, grid_size):
        # Пропускаем через слои и считаем размер
        dummy_input = torch.zeros(1, 1, grid_size, grid_size)  # Исходный размер карты
        x = self.conv1(dummy_input)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x.shape[1]  # Размер после свёрток и пуллинга

    def forward(self, state, grid):
        # Обработка карты препятствий CNN
        x = torch.relu(self.conv1(grid))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)

        # Объединение признаков CNN с остальной информацией о состоянии
        x = torch.cat((state, x), dim=1)

        # Полносвязные слои
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action = self.action_head(x)
        angle = torch.sigmoid(self.angle_head(x)) * 360
        return action, angle
    
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_features, grid, action, reward, next_state_features, next_grid, done, next_angle):
        self.buffer.append((state_features, grid, action, reward, next_state_features, next_grid, done, next_angle))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, grids, actions, rewards, next_states, next_grids, dones, next_angles = zip(*batch)
        return states, grids, actions, rewards, next_states, next_grids, dones, next_angles

    def __len__(self):
        return len(self.buffer)

# Агент DQN
class DQNAgent:
    def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, lr=1e-3):
        self.env = env
        self.size = env.size
        self.state_dim = env.observation_space["agent"].shape[0] + env.observation_space["target"].shape[0] + 2 + 1 # Угол обзора и цель
        self.action_dim = env.action_space["move"].n

        self.q_network = DQN(self.state_dim, self.action_dim, grid_size=self.size).to(device)
        self.target_network = DQN(self.state_dim, self.action_dim, grid_size=self.size).to(device) 
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.update_target_freq = 1000
        self.steps_done = 0
        self.scaler = GradScaler()  # Default initialization


        self.rewards_history = []  # История вознаграждений для графика
        self.wins_history = []  # Счёт выигрышей
        self.losses_history = []  # Счёт проигрышей
        self.lock = threading.Lock()  # Для синхронизации доступа к историям

        # Инициализация графиков и анимации
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.rewards_line, = self.ax1.plot([], label='Current Rewards', color='blue')
        self.ax1.set_xlabel("Episodes")
        self.ax1.set_ylabel("Rewards")
        self.ax1.set_title("Current Rewards")
        self.ax1.legend()

        self.wins_line, = self.ax2.plot([], label='Wins', color='green')
        self.losses_line, = self.ax2.plot([], label='Losses', color='red')
        self.ax2.set_xlabel("Episodes")
        self.ax2.set_ylabel("Count")
        self.ax2.set_title("Wins and Losses")
        self.ax2.legend()

        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            fargs=(self.rewards_history, self.wins_history, self.losses_history, self.rewards_line, self.wins_line, self.losses_line),
            interval=1000,  # Обновление графиков каждую 1 секунду
            blit=True
        )

        # Инициализация DataFrame для записи данных
        self.log_data = pd.DataFrame(columns=['Episode', 'Reward', 'Wins', 'Losses', 'Epsilon'])

        # Функция для обновления графиков
    def update_plot(self, num, rewards_history, wins_history, losses_history, rewards_line, wins_line, losses_line):
        rewards_line.set_data(range(len(rewards_history)), rewards_history)
        wins_line.set_data(range(len(wins_history)), wins_history)
        losses_line.set_data(range(len(losses_history)), losses_history)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        return rewards_line, wins_line, losses_line

    def select_action(self, state_features, grid):  # Добавляем grid
        if random.random() < self.epsilon:
            action = self.env.action_space["move"].sample()
            predicted_angle = random.randint(0, 360)
        else:
            with torch.no_grad():
                action_values, angle_values = self.q_network(state_features, grid)  # Передаем grid
                action = action_values.argmax().item()
                predicted_angle = angle_values.item()

        return action, predicted_angle

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, grids, actions, rewards, next_states, next_grids, dones, next_angles = self.replay_buffer.sample(self.batch_size) 

        states = torch.stack(states).to(device)  # Исправление
        grids = torch.stack(grids).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)  # Исправление
        next_grids = torch.stack(next_grids).to(device)
        dones = torch.FloatTensor(dones).to(device)
        next_angles = torch.FloatTensor(next_angles).to(device)


        # Use the updated autocast context manager with device_type
        with torch.amp.autocast(device_type='cuda'):
            q_values, angles = self.q_network(states, grids)  # Передаем grids
            next_q_values, next_angles = self.target_network(next_states, next_grids)  # Передаем next_grids

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

            loss = huber_loss(expected_q_value, q_value)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > self.epsilon_min:
            if len(self.wins_history) < len(self.losses_history):  # Если проигрышей больше, увеличиваем ε
                self.epsilon += 0.05
            else:
                self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())



    def save_weights(self, episode):
        torch.save(self.q_network.state_dict(), f"dqn_weights_episode_{episode}.pth")
        print(f"Weights saved at episode {episode}")

    def train(self, num_episodes=1000):
        episode_wins = 0
        episode_losses = 0
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                self.env.render()
                state_features, grid = self._flatten_state(state, info)
                action, predicted_angle = self.select_action(state_features, grid) 
                print({"move": action, "view_angle": predicted_angle})
                next_state, reward, done, _, next_info = self.env.step({"move": action, "view_angle": predicted_angle})
                next_state_features, next_grid = self._flatten_state(next_state, next_info) # Исправление 3

                self.replay_buffer.push(state_features, grid, action, reward, next_state_features, next_grid, done, predicted_angle)
                state = next_state
                episode_reward = reward

                self.train_step()

            # Обновление счёта выигрыш/проигрыш
            if episode_reward == 100:
                episode_wins += 1
            elif episode_reward == -100:
                episode_losses += 1

            with self.lock:
                self.rewards_history.append(episode_reward)
                self.wins_history.append(episode_wins)
                self.losses_history.append(episode_losses)

                        # Запись данных в DataFrame
            new_data = pd.DataFrame({
                'Episode': [episode],
                'Reward': [episode_reward],
                'Wins': [episode_wins],
                'Losses': [episode_losses],
                'Epsilon': [self.epsilon]
            })

            self.log_data = pd.concat([self.log_data, new_data], ignore_index=True)

            # Отладочный вывод
            print(f"Episode {episode}: Reward {episode_reward}, Epsilon {self.epsilon}")
            print(f"Current Rewards History: {self.rewards_history}")  # Проверяем историю наград

            #Сохранение таблицы с данными в CSV-файл
            self.log_data.to_csv("training_log_normal_reward_level3.csv", index=False)
            print("Training log saved to training_log.csv")


            # Сохранение весов каждые 100 эпизодов
            if episode % 100 == 0:
                self.save_weights(episode)

            if episode % 10 == 0:
                print(f"Episode {episode}: Reward {episode_reward}, Epsilon {self.epsilon}")

        self.env.close()


    def _flatten_state(self, state, info):
        agent = state["agent"]
        target = state["target"]
        agent_angle = state["agent_angle"]
        target_angle = state["target_angle"]
        distance = info["distance"]
        obs = info["obstacles"]

        # Создание карты препятствий
        grid = np.zeros((self.size, self.size))
        for obstacle in obs:
            grid[obstacle[0], obstacle[1]] = 1
        grid = grid.reshape(1, self.size, self.size)  # Исправление: убираем лишнюю размерность
        grid = torch.FloatTensor(grid).to(device)  # Преобразование в тензор 

        # Остальная информация о состоянии
        state_features = np.concatenate([agent, agent_angle, target, target_angle, [distance]])
        state_features = torch.FloatTensor(state_features).to(device)

        return state_features, grid 
    
# Инициализация среды и агента
env = gym.make("RedAndBlue-v0.1", render_mode=None, size=50, target_behavior='circle')

agent = DQNAgent(env)
agent.train(num_episodes=5000)
