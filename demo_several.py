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
import matplotlib
import multiprocessing
import os
import threading

matplotlib.use('TkAgg')

# Использование GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Сеть для DQN с поддержкой AMP
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# Агент DQN
class DQNAgent:
    def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, lr=1e-3):
        self.env = env
        self.state_dim = env.observation_space["agent"].shape[0] + 2  # Угол обзора и цель
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
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_freq = 1000
        self.steps_done = 0
        self.scaler = GradScaler()  # AMP

        self.rewards_history = []  # История вознаграждений для графика
        self.wins_history = []  # Счёт выигрышей
        self.losses_history = []  # Счёт проигрышей

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space["move"].sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        with autocast():
            q_values = self.q_network(states)
            next_q_values = self.target_network(next_states)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

            loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, num_episodes=1000):
        episode_wins = 0
        episode_losses = 0
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(self._flatten_state(state))
                next_state, reward, done, _, _ = self.env.step({"move": action, "view_angle": random.randint(0, 360)})

                self.replay_buffer.push(self._flatten_state(state), action, reward, self._flatten_state(next_state), done)
                state = next_state
                episode_reward += reward

                self.train_step()

            # Обновление счёта выигрыш/проигрыш
            if episode_reward == 100:
                episode_wins += 1
            elif episode_reward == -100:
                episode_losses += 1

            self.rewards_history.append(episode_reward)
            self.wins_history.append(episode_wins)
            self.losses_history.append(episode_losses)

            # Отладочный вывод
            print(f"Episode {episode}: Reward {episode_reward}, Epsilon {self.epsilon}")

        self.save_weights("dqn_weights.pth")  # Сохранение весов после обучения
        self.env.close()

    def save_weights(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Weights saved to {filepath}")

    def _flatten_state(self, state):
        agent = state["agent"]
        agent_angle = state["agent_angle"]
        target_angle = state["target_angle"]
        return np.concatenate([agent, agent_angle, target_angle])

def train_agent_in_env(env_name, num_episodes, agent_id):
    env = gym.make(env_name, render_mode=None, size=50, target_behavior='circle')
    agent = DQNAgent(env)

    # Запускаем поток для обновления графика в реальном времени
    plot_thread = threading.Thread(target=update_plot, args=(agent_id, agent))
    plot_thread.start()

    agent.train(num_episodes=num_episodes)

    # Завершаем поток после завершения обучения
    plot_thread.join()

def update_plot(agent_id, agent):
    plt.ion()  # Включаем интерактивный режим
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Настройки графиков
    axs[0].set_title(f'Agent {agent_id} - Wins/Losses')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Count')
    
    axs[1].set_title(f'Agent {agent_id} - Rewards')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Reward')

    while True:
        # Проверяем, завершилось ли обучение
        if len(agent.rewards_history) == 0:
            continue

        # Очищаем графики
        axs[0].cla()
        axs[1].cla()

        # Отрисовка графиков выигрышей и проигрышей
        axs[0].plot(agent.wins_history, label='Wins', color='g')
        axs[0].plot(agent.losses_history, label='Losses', color='r')
        axs[0].legend()

        # Отрисовка графиков наград
        axs[1].plot(agent.rewards_history, label='Rewards', color='b')
        axs[1].legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Пауза для обновления графиков

        # Если обучение завершено, выходим из цикла
        if len(agent.wins_history) >= agent.train_steps:
            break

if __name__ == "__main__":
    # Конфигурация агентов
    agents_config = [
        {"env_name": "RedAndBlue-v0.1", "num_episodes": 500, "buffer_size": 100000, "batch_size": 256, "gamma": 0.99, "lr": 1e-3},
        {"env_name": "RedAndBlue-v0.1", "num_episodes": 500, "buffer_size": 200000, "batch_size": 128, "gamma": 0.95, "lr": 5e-4},
        {"env_name": "RedAndBlue-v0.1", "num_episodes": 500, "buffer_size": 150000, "batch_size": 64, "gamma": 0.90, "lr": 1e-3},
        {"env_name": "RedAndBlue-v0.1", "num_episodes": 500, "buffer_size": 250000, "batch_size": 512, "gamma": 0.99, "lr": 1e-4},
    ]

    num_processes = len(agents_config)  # Количество процессов
    processes = []
    for agent_id, config in enumerate(agents_config):
        p = multiprocessing.Process(target=train_agent_in_env, args=(config["env_name"], config["num_episodes"], agent_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
