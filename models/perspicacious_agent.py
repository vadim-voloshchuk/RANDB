import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

class PretentiousAgent:
    def __init__(self, grid_size=100, state_size=5, move_actions=5, view_actions=12, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, sequence_length=10, model_path=None):

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        if model_path is not None and os.path.exists(model_path):
            self.load_weights(model_path)
        self.state_sequence = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        """Builds the CNN-RNN model with two outputs."""
        state_input = nn.Linear(self.sequence_length * self.state_size, 64).to(self.device)
        grid_input = nn.Conv2d(1, 32, kernel_size=3, padding=1).to(self.device)
        grid_input = nn.ReLU().to(self.device)
        grid_input = nn.MaxPool2d(kernel_size=2).to(self.device)
        grid_input = nn.Flatten().to(self.device)

        lstm = nn.LSTM(input_size=self.state_size, hidden_size=64, batch_first=True).to(self.device)

        merged = nn.Linear(64 + grid_input.out_features, 128).to(self.device)
        merged = nn.ReLU().to(self.device)

        move_output = nn.Linear(128, self.move_actions).to(self.device)
        view_output = nn.Linear(128, self.view_actions).to(self.device)

        model = nn.ModuleList([state_input, grid_input, lstm, merged, move_output, view_output])
        return model

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
                move_q_values, view_q_values = self._predict(input_sequence, grid_input)
                move_action = torch.argmax(move_q_values).item()
                view_action = torch.argmax(view_q_values).item()
            else:
                padding_length = self.sequence_length - len(self.state_sequence)
                padding = np.zeros((padding_length, self.state_size))
                padded_sequence = np.concatenate((padding, self.state_sequence))
                input_sequence = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                grid_input = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                move_q_values, view_q_values = self._predict(input_sequence, grid_input)
                move_action = torch.argmax(move_q_values).item()
                view_action = torch.argmax(view_q_values).item()

        return move_action, view_action

    def _predict(self, state_input, grid_input):
        state_input = self.model[0](state_input.view(-1, self.sequence_length * self.state_size))
        grid_input = self.model[1](grid_input)
        grid_input = grid_input.view(grid_input.size(0), -1)
        lstm_out, _ = self.model[2](state_input.unsqueeze(0))
        merged = torch.cat((lstm_out[:, -1, :], grid_input), dim=1)
        merged = self.model[3](merged)
        move_q_values = self.model[4](merged)
        view_q_values = self.model[5](merged)
        return move_q_values, view_q_values

    def train(self, state, move_action, view_action, reward, next_state, done, grid, next_grid):
        """Trains the agent using separate targets for move and view angle."""
        target_move = reward
        target_view = reward

        if not done:
            next_state_sequence = self.state_sequence + [next_state[0]]
            if len(next_state_sequence) > self.sequence_length:
                next_state_sequence.pop(0)

            if len(next_state_sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(next_state_sequence)
                padding = np.zeros((padding_length, self.state_size))
                next_state_sequence = np.concatenate((padding, next_state_sequence))

            next_state_sequence = np.array([next_state_sequence])
            next_grid_input = np.expand_dims(next_grid, axis=0)

            if len(next_state_sequence[0]) == self.sequence_length:
                next_move_q_values, next_view_q_values = self._predict(torch.tensor(next_state_sequence, dtype=torch.float32).to(self.device), torch.tensor(next_grid_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device))
                target_move = reward + self.gamma * torch.max(next_move_q_values).item()
                target_view = reward + self.gamma * torch.max(next_view_q_values).item()

        if len(self.state_sequence) == self.sequence_length:
            input_sequence = torch.tensor(self.state_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            grid_input = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            target_move_f, target_view_f = self._predict(input_sequence, grid_input)
            target_move_f[0][move_action] = target_move
            target_view_f[0][view_action] = target_view
            loss = self._calculate_loss(target_move_f, target_view_f, move_action, view_action)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _calculate_loss(self, move_q_values, view_q_values, move_action, view_action):
        move_loss = nn.MSELoss()(move_q_values, torch.eye(self.move_actions)[move_action].unsqueeze(0).to(self.device))
        view_loss = nn.MSELoss()(view_q_values, torch.eye(self.view_actions)[view_action].unsqueeze(0).to(self.device))
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
                print(reward, state. next_state)
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
        distance_to_target = np.linalg.norm(agent_pos - target_pos)
        processed_state = np.concatenate((agent_pos, target_pos, [distance_to_target]))
        return processed_state

    def _process_action(self, move_action, view_action):
        view_angle_action = view_action * 30  # Example: 12 actions -> 360 degrees
        return {"move": move_action, "view_angle": view_angle_action}

    def _create_grid_representation(self, obstacles):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obstacle in obstacles:
            grid[obstacle[0], obstacle[1]] = 1.0
        return grid

    def load_weights(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Weights loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading weights from {model_path}: {e}")

    def save_weights(self, model_path):
        try:
            torch.save(self.model.state_dict(), model_path)
            print(f"Weights saved successfully to: {model_path}")
        except Exception as e:
            print(f"Error saving weights to {model_path}: {e}")