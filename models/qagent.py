import numpy as np

class QLearningAgent:
    def __init__(self, env, grid_size=100, move_actions=5, view_actions=12,
                 learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):

        self.env = env
        self.grid_size = grid_size
        self.move_actions = move_actions
        self.view_actions = view_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((grid_size, grid_size, view_actions, move_actions))

    def _get_state_index(self, state):
        agent_x, agent_y = state["agent"]
        return agent_x, agent_y

    def choose_action(self, state):
        state_index = self._get_state_index(state)
        if np.random.rand() <= self.epsilon:
            move_action = np.random.randint(self.move_actions)
            view_action = np.random.randint(self.view_actions)
        else:
            move_action = np.argmax(self.q_table[state_index[0], state_index[1], :, :])
            view_action = np.argmax(self.q_table[state_index[0], state_index[1], move_action, :])
        return move_action, view_action

    def train(self, state, move_action, view_action, reward, next_state, done):
        state_index = self._get_state_index(state)
        next_state_index = self._get_state_index(next_state)

        if not done:
            next_max_q = np.max(self.q_table[next_state_index[0], next_state_index[1], :, :])
            target_q = reward + self.gamma * next_max_q
        else:
            target_q = reward

        self.q_table[state_index[0], state_index[1], view_action, move_action] += self.learning_rate * (target_q - self.q_table[state_index[0], state_index[1], view_action, move_action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_model(self, env, output_dir="./", episodes=1000):
        results = []
        wins = 0
        loses = 0
        for episode in range(episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            iteration = 0

            while not done:
                move_action, view_action = self.choose_action(state)
                env_action = self._process_action(move_action, view_action)
                next_state, reward, done, _, info = env.step(env_action)
                self.train(state, move_action, view_action, reward, next_state, done)
                state = next_state
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

    def _process_action(self, move_action, view_action):
        """Maps actions to environment format. View angle is discretized."""
        view_angle_action = view_action * 30  # Example: 12 actions -> 360 degrees
        return {"move": move_action, "view_angle": view_angle_action}