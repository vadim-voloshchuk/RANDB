import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random
from collections import deque

class PretentiousAgent:
    def __init__(self, grid_size=100, state_size=5, move_actions=5, view_actions=12, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, sequence_length=10, replay_buffer_size=10000, 
                 batch_size=64, model_path=None):

        self.grid_size = grid_size
        self.state_size = state_size
        self.move_actions = move_actions
        self.view_actions = view_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sequence_length = sequence_length
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()  # Copy weights from model to target_model

        if model_path is not None and os.path.exists(model_path):
            self.load_weights(model_path)

        self.state_sequence = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        print("Building model...")
        try:
            state_input = nn.Sequential(
                nn.Linear(self.sequence_length * self.state_size, 128),
                nn.ReLU()
            ).to(self.device)
            print("State input created.")
            
            conv_layers = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten()
            ).to(self.device)
            print("Convolutional layers created.")
            
            lstm = nn.LSTM(input_size=self.state_size, hidden_size=128, batch_first=True).to(self.device)
            print("LSTM layer created.")

            merge_layer = nn.Sequential(
                nn.Linear(128 + 64 * (self.grid_size // 2) * (self.grid_size // 2), 256),
                nn.ReLU()
            ).to(self.device)
            print("Merge layer created.")

            move_output = nn.Linear(256, self.move_actions).to(self.device)
            view_output = nn.Linear(256, self.view_actions).to(self.device)
            print("Output layers created.")

            model = nn.ModuleList([state_input, conv_layers, lstm, merge_layer, move_output, view_output])
            print("Model built successfully.")
            return model
        except Exception as e:
            print(f"Error during model building: {e}")


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, grid):
        """Chooses move and view angle actions using the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            move_action = np.random.randint(self.move_actions)
            view_action = np.random.randint(self.view_actions)
        else:
            self.state_sequence.append(state[0])
            if len(self.state_sequence) > self.sequence_length:
                self.state_sequence.pop(0)

            if len(self.state_sequence) == self.sequence_length:
                input_sequence = torch.tensor(self.state_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                grid_input = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                move_q_values, view_q_values = self._predict(input_sequence, grid_input, use_target=False)
                move_action = torch.argmax(move_q_values).item()
                view_action = torch.argmax(view_q_values).item()
            else:
                padding_length = self.sequence_length - len(self.state_sequence)
                padding = np.zeros((padding_length, self.state_size))
                padded_sequence = np.concatenate((padding, self.state_sequence))
                input_sequence = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                grid_input = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                move_q_values, view_q_values = self._predict(input_sequence, grid_input, use_target=False)
                move_action = torch.argmax(move_q_values).item()
                view_action = torch.argmax(view_q_values).item()

        return move_action, view_action

    def _predict(self, state_input, grid_input, use_target=False):
        model = self.target_model if use_target else self.model
        state_input = model[0](state_input.view(-1, self.sequence_length * self.state_size))
        grid_input = model[1](grid_input)
        lstm_out, _ = model[2](state_input.unsqueeze(0))
        merged = torch.cat((lstm_out[:, -1, :], grid_input), dim=1)
        merged = model[3](merged)
        move_q_values = model[4](merged)
        view_q_values = model[5](merged)
        return move_q_values, view_q_values

    def train(self, state, move_action, view_action, reward, next_state, done, grid, next_grid):
        """Stores transition in replay buffer and trains the agent if enough samples."""
        self.replay_buffer.append((state, move_action, view_action, reward, next_state, done, grid, next_grid))

        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            self._train_batch(batch)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _train_batch(self, batch):
        """Train the model on a batch of experiences."""
        states, move_actions, view_actions, rewards, next_states, dones, grids, next_grids = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        grids = torch.tensor(grids, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        next_grids = torch.tensor(next_grids, dtype=torch.float32).unsqueeze(1).to(self.device)

        move_q_values, view_q_values = self._predict(states, grids)
        move_targets = move_q_values.clone().detach()
        view_targets = view_q_values.clone().detach()

        next_move_q_values, next_view_q_values = self._predict(next_states, next_grids, use_target=True)

        for i in range(self.batch_size):
            move_targets[i, move_actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_move_q_values[i]).item()
            view_targets[i, view_actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_view_q_values[i]).item()

        loss = self._calculate_loss(move_q_values, view_q_values, move_targets, view_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calculate_loss(self, move_q_values, view_q_values, move_targets, view_targets):
        move_loss = nn.MSELoss()(move_q_values, move_targets)
        view_loss = nn.MSELoss()(view_q_values, view_targets)
        return move_loss + view_loss

    def run_model(self, env, output_dir="./", episodes=1000):
        results = []
        wins = 0
        loses = 0
        for episode in range(episodes):
            state, info = env.reset()
            state = self._process_state(state)
            state = np.reshape(state, [1, self.state_size])
            grid = self._create_grid_representation(info["obstacles"])
            self.state_sequence = []
            done = False
            total_reward = 0
            iteration = 0

            while not done:
                move_action, view_action = self.choose_action(state, grid)
                env_action = self._process_action(move_action, view_action)
                next_state, reward, done, _, info = env.step(env_action)
                next_state = self._process_state(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                next_grid = self._create_grid_representation(info["obstacles"])
                self.train(state, move_action, view_action, reward, next_state, done, grid, next_grid)
                print(reward, state, next_state)
                state = next_state
                grid = next_grid
                total_reward += reward
                iteration += 1

            if total_reward > 0:
                wins += 1
            else:
                loses += 1

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            results.append({"episode": episode + 1, "total_reward": total_reward, "wins": wins, "loses": loses})

        env.close()
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        self.save_weights("./cnn_rnn_model_2out.weights.pth")

    def _process_state(self, state):
        agent_pos = state["agent"]
        target_pos = state["target"]
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(agent_pos))
        return [agent_pos[0], agent_pos[1], target_pos[0], target_pos[1], distance_to_target]

    def _create_grid_representation(self, obstacles):
        grid = np.zeros((self.grid_size, self.grid_size))
        for (x, y) in obstacles:
            grid[x, y] = 1
        return grid

    def _process_action(self, move_action, view_action):
        return {"move": move_action, "view": view_action}

    def save_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()
