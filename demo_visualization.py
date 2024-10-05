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
from torch.cuda.amp import autocast, GradScaler  # For using AMP
import threading
import matplotlib

matplotlib.use('TkAgg')

# Using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Network with AMP support
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

# DQN Agent
class DQNAgent:
    def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, lr=1e-3):
        self.env = env
        self.state_dim = env.observation_space["agent"].shape[0] + 2  # Viewing angle and target
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

        self.rewards_history = []  # Rewards history for plotting
        self.wins_history = []  # Wins history for plotting
        self.losses_history = []  # Losses history for plotting
        self.lock = threading.Lock()  # For synchronizing access to histories
        self.plot_thread = threading.Thread(target=self._live_plot_rewards, daemon=True)
        self.plot_thread.start()

    def _live_plot_rewards(self):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        rewards_line, = ax.plot([], label='Total Rewards')
        wins_line, = ax.plot([], label='Wins', color='green')
        losses_line, = ax.plot([], label='Losses', color='red')
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Counts")
        ax.set_title("Training Progress")
        ax.legend()

        while True:
            with self.lock:  # Protect access to history
                if self.rewards_history or self.wins_history or self.losses_history:  # Update if there is data
                    rewards_line.set_ydata(self.rewards_history)
                    rewards_line.set_xdata(range(len(self.rewards_history)))
                    wins_line.set_ydata(self.wins_history)
                    wins_line.set_xdata(range(len(self.wins_history)))
                    losses_line.set_ydata(self.losses_history)
                    losses_line.set_xdata(range(len(self.losses_history)))
                    ax.relim()  # Recalculate axis limits
                    ax.autoscale_view()  # Auto scale axes

            plt.draw()
            plt.pause(0.1)  # Pause for plot update

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
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_wins = 0
            episode_losses = 0

            while not done:
                self.env.render()
                action = self.select_action(self._flatten_state(state))
                next_state, reward, done, _, _ = self.env.step({"move": action, "view_angle": random.randint(0, 360)})

                self.replay_buffer.push(self._flatten_state(state), action, reward, self._flatten_state(next_state), done)
                state = next_state
                episode_reward += reward

                self.train_step()

            # Update win/loss counts
            if episode_reward == 100:
                episode_wins = 1
            elif episode_reward == -100:
                episode_losses = 1

            with self.lock:
                self.rewards_history.append(episode_reward)
                self.wins_history.append(episode_wins)
                self.losses_history.append(episode_losses)

            # Debug output
            print(f"Episode {episode}: Reward {episode_reward}, Epsilon {self.epsilon}")
            print(f"Current Rewards History: {self.rewards_history}")  # Check rewards history

            if episode % 10 == 0:
                print(f"Episode {episode}: Reward {episode_reward}, Epsilon {self.epsilon}")

        self.env.close()

    def _flatten_state(self, state):
        agent = state["agent"]
        agent_angle = state["agent_angle"]
        target_angle = state["target_angle"]
        return np.concatenate([agent, agent_angle, target_angle])

# Initialize environment and agent
env = gym.make("RedAndBlue-v0.1", render_mode=None, size=50, target_behavior='circle')

agent = DQNAgent(env)
agent.train(num_episodes=500)
