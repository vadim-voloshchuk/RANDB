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
    def __init__(self, input_dim, output_dim, angle_dim=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Увеличено количество нейронов
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Третий слой увеличен
        self.action_head = nn.Linear(128, output_dim)
        self.angle_head = nn.Linear(128, angle_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # Применяем активацию для третьего слоя
        action = self.action_head(x)
        angle = torch.sigmoid(self.angle_head(x)) * 360  # Масштабирование угла до диапазона [0, 360]
        return action, angle

    
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_angle):
        self.buffer.append((state, action, reward, next_state, done, next_angle))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_angles = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones, next_angles

    def __len__(self):
        return len(self.buffer)

# Агент DQN
class DQNAgent:
    def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, lr=1e-3):
        self.env = env
        self.size = env.size
        self.state_dim = env.observation_space["agent"].shape[0] + env.observation_space["target"].shape[0] + 2 + 1  + self.size * self.size# Угол обзора и цель
        self.action_dim = env.action_space["move"].n

        self.q_network = DQN(self.state_dim, self.action_dim).to(device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(device)
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

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space["move"].sample()
            predicted_angle = random.randint(0, 360)  # Случайный угол для инициализации
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values, angle_values = self.q_network(state)
                action = action_values.argmax().item()
                predicted_angle = angle_values.item()  # Получаем индекс угла

        return action, predicted_angle

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, next_angles = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        next_angles = torch.FloatTensor(next_angles).to(device)

        # Use the updated autocast context manager with device_type
        with torch.amp.autocast(device_type='cuda'):
            q_values, angles = self.q_network(states)
            next_q_values, next_angles = self.target_network(next_states)

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
                action, predicted_angle = self.select_action(self._flatten_state(state, info))
                next_state, reward, done, _, next_info = self.env.step({"move": action, "view_angle": predicted_angle})
                print({"reward": reward, "move": action, "view_angle": predicted_angle, "w": next_state["agent_w"], "l": next_state["agent_l"]})

                self.replay_buffer.push(self._flatten_state(state, info), action, reward, self._flatten_state(next_state, next_info), done, predicted_angle)
                state = next_state
                episode_reward = reward

                self.train_step()

            # Обновление счёта выигрыш/проигрыш
            if next_state["agent_w"]:
                episode_wins += 1
            elif next_state["agent_l"]:
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
            self.log_data.to_csv("training_log_normal_reward_level2.csv", index=False)
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
        
        # Инициализируем матрицу 100x100 (например, с препятствиями)
        grid = np.zeros((self.size, self.size))
        for obstacle in obs:
            grid[obstacle[0], obstacle[1]] = 1

        flattened_grid = grid.flatten()

        # Добавляем матрицу к остальным частям состояния
        return np.concatenate([agent, agent_angle, target, target_angle, [distance], flattened_grid])

# Инициализация среды и агента
env = gym.make("RedAndBlue-v0.1", render_mode=None, size=50, target_behavior='circle')

agent = DQNAgent(env)
agent.train(num_episodes=5000)
