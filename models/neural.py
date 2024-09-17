import numpy as np
import os
import json
import tensorflow as tf

class NeuralAgent:
    def __init__(self, state_size=5, action_size=5, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=self.action_size, activation='linear')  # Linear output for Q-values
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])  # Exploit

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_model(self, env, output_dir = "./", episodes=1000):
        results = []
        for episode in range(episodes):
            state, info = env.reset()
            # Adapt state representation
            state = self._process_state(state)
            state = np.reshape(state, [1, self.state_size])  # Reshape for the network
            done = False
            total_reward = 0
            iteration = 0

            while not done:
                action = self.choose_action(state)
                # Adapt action to environment's format
                env_action = self._process_action(action)
                print(env_action, iteration)
                next_state, reward, done, _, _ = env.step(env_action)
                next_state = self._process_state(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.train(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                iteration += 1

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            results.append({"episode": episode + 1, "total_reward": total_reward})

        env.close()

        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)


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
    
    