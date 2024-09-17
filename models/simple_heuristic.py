import numpy as np
import os
import json


class HeuristicAgent:
    def __init__(self):
        self._agent_old_position = np.array([0, 0])
        self._stuck = 0
        self._stuck_path = 0
        self.view_distance = 10
        self.view_angle = 90

    def set_init_position(self, observation, info):
        self._agent_old_position = observation["agent"]
        self._stuck = 0
        self._stuck_path = 0

    def choose_action(self, observation, info):
        agent_pos = observation["agent"]
        target_pos = observation["target"]
        agent_angle = observation['agent_angle']
        target_angle = observation['target_angle']
        distance_to_target = np.linalg.norm(agent_pos - target_pos)

        if (distance_to_target == self.view_distance + 1) and self._is_in_view(agent_pos,target_pos,target_angle):
            return {"move": 4, "view_angle": agent_angle[0]}

        if np.array_equal(agent_pos, self._agent_old_position):
            self._stuck += 1
        else:
            self._agent_old_position = agent_pos
            self._stuck = 0

        if self._stuck >= 5:
            self._stuck_path = self._stuck

        move_action = self._move_towards_target(agent_pos, target_pos)
        view_angle_action = self._adjust_view_angle(agent_pos, target_pos, agent_angle)

        return {"move": move_action, "view_angle": view_angle_action}

    def _move_towards_target(self, agent_pos, target_pos):
        direction = np.sign(target_pos - agent_pos)

        if self._stuck_path >= 0:
            self._stuck_path -= 1
            return np.random.randint(0, 4)

        if direction[0] == 1:
            return 0
        elif direction[1] == 1:
            return 1
        elif direction[0] == -1:
            return 2
        elif direction[1] == -1:
            return 3
        else:
            return 4

    def _adjust_view_angle(self, agent_pos, target_pos, agent_angle):
        delta = target_pos - agent_pos
        target_angle = (np.degrees(np.arctan2(delta[1], delta[0])) + 360) % 360
        return int(target_angle)

    def _is_in_view(self, agent_pos, target_pos, target_angle):
        delta = agent_pos - target_pos
        angle_to_agent = (np.degrees(np.arctan2(delta[1], delta[0])) - target_angle) % 360

        if angle_to_agent > 180:
            angle_to_agent -= 360

        in_view = -self.view_angle // 2 <= angle_to_agent <= self.view_angle // 2
        return in_view

    def run_model(self, env, episodes, output_dir):
        reward = None
        steps = 10
        win = 0
        lose = 0
        results = []

        for episode in range(episodes):
            observation, info = env.reset()
            self.set_init_position(observation, info)
            total_reward = 0

            done = False
            while not done:
                action = self.choose_action(observation, info)
                observation, reward, done, _, info = env.step(action)
                total_reward += reward

            if total_reward:
                win += 1
            else:
                lose += 1

            print(f"Win:{win} , Lose:{lose}")
            results.append({"episode": episode + 1, "total_reward": total_reward, "wins": win, "loses": lose})


        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        env.close()
