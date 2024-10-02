import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeAgent:
    def __init__(self, env, grid_size=100, move_actions=5, view_actions=12):
        self.env = env
        self.grid_size = grid_size
        self.move_actions = move_actions
        self.view_actions = view_actions

        self.move_model = DecisionTreeClassifier()
        self.view_model = DecisionTreeClassifier()

        self.state_history = []
        self.move_action_history = []
        self.view_action_history = []
        self.reward_history = []

    def _get_features(self, state, grid):
        agent_x, agent_y = state["agent"]
        target_x, target_y = state["target"]
        features = [agent_x, agent_y, target_x, target_y]
        return features

    def choose_action(self, state, grid):
        features = self._get_features(state, grid)
        move_action = self.move_model.predict([features])[0]
        view_action = self.view_model.predict([features])[0]
        return move_action, view_action

    def train(self, state, move_action, view_action, reward, next_state, done, grid):
        self.state_history.append(self._get_features(state, grid))
        self.move_action_history.append(move_action)
        self.view_action_history.append(view_action)
        self.reward_history.append(reward)

        if done:
            self.move_model.fit(self.state_history, self.move_action_history)
            self.view_model.fit(self.state_history, self.view_action_history)
            self.state_history = []
            self.move_action_history = []
            self.view_action_history = []
            self.reward_history = []

    def run_model(self, env, output_dir="./", episodes=1000):
        results = []
        wins = 0
        loses = 0
        for episode in range(episodes):
            state, info = env.reset()
            grid = self._create_grid_representation(info["obstacles"])
            done = False
            total_reward = 0
            iteration = 0

            while not done:
                move_action, view_action = self.choose_action(state, grid)
                env_action = self._process_action(move_action, view_action)
                next_state, reward, done, _, info = env.step(env_action)
                self.train(state, move_action, view_action, reward, next_state, done, grid)
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

    def _create_grid_representation(self, obstacles):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obstacle in obstacles:
            grid[obstacle[0], obstacle[1]] = 1.0
        return grid