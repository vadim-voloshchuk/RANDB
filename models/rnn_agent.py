import numpy as np
import tensorflow as tf
import os
import json

class RNNNeuralAgent:
    def __init__(self, state_size=5, action_size=5, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, sequence_length=10, model_path = None): 
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sequence_length = sequence_length  # Length of state sequence for LSTM

        self.model = self._build_model()
        self.model = self._build_model()
        if model_path is not None and os.path.exists(model_path):  # Load weights if path is provided
            self.load_weights(model_path)
        self.state_sequence = [] 

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=64, input_shape=(self.sequence_length, self.state_size)),
            tf.keras.layers.Dense(units=self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            self.state_sequence.append(state[0])
            if len(self.state_sequence) > self.sequence_length:
                self.state_sequence.pop(0)

            if len(self.state_sequence) == self.sequence_length:
                input_sequence = np.array([self.state_sequence])
                q_values = self.model.predict(input_sequence)
                return np.argmax(q_values[0])
            else:
                # Pad the sequence with zeros at the beginning
                padding_length = self.sequence_length - len(self.state_sequence)
                padding = np.zeros((padding_length, self.state_size)) 
                padded_sequence = np.concatenate((padding, self.state_sequence))  # Pad at the beginning
                input_sequence = np.array([padded_sequence])
                q_values = self.model.predict(input_sequence)
                return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state_sequence = self.state_sequence + [next_state[0]]
            if len(next_state_sequence) > self.sequence_length:
                next_state_sequence.pop(0)

            if len(next_state_sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(next_state_sequence)
                padding = np.zeros((padding_length, self.state_size))
                next_state_sequence = np.concatenate((padding, next_state_sequence))  # Pad at the beginning

            next_state_sequence = np.array([next_state_sequence])

            # Calculate target ONLY if not done AND sequence length is correct
            if len(next_state_sequence[0]) == self.sequence_length:
                target = reward + self.gamma * np.amax(self.model.predict(next_state_sequence)[0])

        # Prepare the current state sequence for training
        if len(self.state_sequence) == self.sequence_length:
            input_sequence = np.array([self.state_sequence])
            target_f = self.model.predict(input_sequence)
            target_f[0][action] = target
            self.model.fit(input_sequence, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def run_model(self, env, output_dir = "./", episodes=1000):
        results = []
        wins = 0
        loses = 0
        for episode in range(episodes):
            state, info = env.reset()
            state = self._process_state(state)
            state = np.reshape(state, [1, self.state_size])
            self.state_sequence = []  # Reset state sequence at the beginning of each episode 
            done = False
            total_reward = 0
            iteration = 0

            while not done:
                action = self.choose_action(state)
                env_action = self._process_action(action)
                next_state, reward, done, _, _ = env.step(env_action)
                next_state = self._process_state(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.train(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                iteration += 1

            if total_reward > 0:
                wins += 1
            else:
                loses +=1

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            results.append({"episode": episode + 1, "total_reward": total_reward, "wins": wins, "loses": loses})

        env.close()
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        self.save_weights("./rnn_model.weights.h5")

    def _process_state(self, state):
        """
        Adapt the state representation from the environment to the agent's input format.
        """
        agent_pos = state["agent"]
        target_pos = state["target"]
        distance_to_target = np.linalg.norm(agent_pos - target_pos)

        # Combine relevant state features (you might need to adjust this based on your needs)
        processed_state = np.concatenate((agent_pos, target_pos, [distance_to_target])) 
        return processed_state

    def _process_action(self, action):
        """
        Adapt the agent's action to the environment's action format.
        """
        # Simple mapping for now (you might need a more complex mapping)
        move_action = action # 0-4 for movement
        view_angle_action = action * 60  # Discretize view angle to multiples of 45 degrees 

        return {"move": move_action, "view_angle": view_angle_action}
    
    def load_weights(self, model_path):
        """Loads weights from a saved model file."""
        try:
            self.model.load_weights(model_path)
            print(f"Weights loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading weights from {model_path}: {e}")

    def save_weights(self, model_path):
        """Saves the model's weights to a file."""
        try:
            self.model.save_weights(model_path)
            print(f"Weights saved successfully to: {model_path}")
        except Exception as e:
            print(f"Error saving weights to {model_path}: {e}")
