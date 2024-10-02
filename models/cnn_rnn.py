import numpy as np
import tensorflow as tf
import os
import json

class CNNRNNNeuralAgent:
    def __init__(self, grid_size=100, state_size=5, move_actions=5, view_actions=12, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, sequence_length=10, model_path=None):

        self.grid_size = grid_size
        self.state_size = state_size
        self.move_actions = move_actions  # Number of possible move actions
        self.view_actions = view_actions # Number of possible view angle actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sequence_length = sequence_length

        self.model = self._build_model()
        if model_path is not None and os.path.exists(model_path):
            self.load_weights(model_path)
        self.state_sequence = []

    def _build_model(self):
        """Builds the CNN-RNN model with two outputs."""
        state_input = tf.keras.layers.Input(shape=(self.sequence_length, self.state_size))
        grid_input = tf.keras.layers.Input(shape=(self.grid_size, self.grid_size, 1))

        # CNN for processing grid information
        cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(grid_input)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = tf.keras.layers.Flatten()(cnn)

        # RNN for processing state sequence
        lstm = tf.keras.layers.LSTM(units=64, input_shape=(self.sequence_length, self.state_size))(state_input)

        # Concatenate CNN and RNN outputs
        merged = tf.keras.layers.Concatenate()([lstm, cnn])

        # Separate output layers for move and view angle actions
        move_output = tf.keras.layers.Dense(units=self.move_actions, activation='linear', name='move_output')(merged)
        view_output = tf.keras.layers.Dense(units=self.view_actions, activation='linear', name='view_output')(merged)

        model = tf.keras.models.Model(inputs=[state_input, grid_input], outputs=[move_output, view_output])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='mse')
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
                input_sequence = np.array([self.state_sequence])
                grid_input = np.expand_dims(grid, axis=0)
                move_q_values, view_q_values = self.model.predict([input_sequence, grid_input])
                move_action = np.argmax(move_q_values[0])
                view_action = np.argmax(view_q_values[0])
            else:
                padding_length = self.sequence_length - len(self.state_sequence)
                padding = np.zeros((padding_length, self.state_size))
                padded_sequence = np.concatenate((padding, self.state_sequence))
                input_sequence = np.array([padded_sequence])
                grid_input = np.expand_dims(grid, axis=0)
                move_q_values, view_q_values = self.model.predict([input_sequence, grid_input])
                move_action = np.argmax(move_q_values[0])
                view_action = np.argmax(view_q_values[0])

        return move_action, view_action

    def train(self, state, move_action, view_action, reward, next_state, done, grid, next_grid):
        """Trains the agent using separate targets for move and view angle."""
        target_move = reward
        target_view = reward # You might want separate reward logic for view angle

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
                next_move_q_values, next_view_q_values = self.model.predict([next_state_sequence, next_grid_input])
                target_move = reward + self.gamma * np.amax(next_move_q_values[0])
                target_view = reward + self.gamma * np.amax(next_view_q_values[0])

        if len(self.state_sequence) == self.sequence_length:
            input_sequence = np.array([self.state_sequence])
            grid_input = np.expand_dims(grid, axis=0)
            target_move_f, target_view_f = self.model.predict([input_sequence, grid_input])
            target_move_f[0][move_action] = target_move
            target_view_f[0][view_action] = target_view
            self.model.fit([input_sequence, grid_input], [target_move_f, target_view_f], epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

        self.save_weights("./cnn_rnn_model_2out.weights.h5")

    def _process_state(self, state):
        agent_pos = state["agent"]
        target_pos = state["target"]
        distance_to_target = np.linalg.norm(agent_pos - target_pos)
        processed_state = np.concatenate((agent_pos, target_pos, [distance_to_target]))
        return processed_state

    def _process_action(self, move_action, view_action):
        """Maps actions to environment format. View angle is discretized."""
        view_angle_action = view_action * 30  # Example: 12 actions -> 360 degrees
        return {"move": move_action, "view_angle": view_angle_action}

    def _create_grid_representation(self, obstacles):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obstacle in obstacles:
            grid[obstacle[0], obstacle[1]] = 1.0
        return grid
    
    def load_weights(self, model_path):
        try:
            self.model.load_weights(model_path)
            print(f"Weights loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading weights from {model_path}: {e}")

    def save_weights(self, model_path):
        try:
            self.model.save_weights(model_path)
            print(f"Weights saved successfully to: {model_path}")
        except Exception as e:
            print(f"Error saving weights to {model_path}: {e}")