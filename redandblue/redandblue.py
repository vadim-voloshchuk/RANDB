import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame

import math


def _bresenham(start, end):
    points = []
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points


class RedAndBlue(gym.Env):
    metadata = {'render_modes': ['human'],
                'render_fps': 4,
                'obstacle_types': ['random', 'preset'],
                'target_behaviors': ['random', 'circle']}

    obstacle_presets = [
        [(0, 0), (0, 1), (0, 2), (1, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 3)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)],
        [(1, 1), (1, 0), (1, 2), (0, 1), (2, 1), (3, 3)],
    ]

    min_obstacles = 5
    max_obstacles = 7

    min_obstacles_len = 1
    max_obstacles_len = 15

    view_distance = 10
    view_angle = 90

    max_steps_per_round = 200  # Maximum steps allowed before the round ends
    negative_reward_threshold = -50  # Threshold for negative reward to trigger a loss


    def __init__(self, render_mode=None, size=100, target_behavior='circle', obstacle_percentage=0.2, obstacle_type='random', fps=4):
        self._agent_location = None
        self._target_location = None
        self.size = size
        self.window_size = 1000
        self.metadata["render_fps"] = fps
        self.obstacle_percentage = obstacle_percentage
        self.obstacle_type = obstacle_type
        self.target_behavior = target_behavior

        self.agent_angle = 0
        self.target_angle = 0

        self.action_space = spaces.Dict({
            "move": spaces.Discrete(5),
            "view_angle": spaces.Discrete(360)
        })

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "agent_angle": spaces.Box(0, 360, shape=(1,), dtype=int),
                "target_angle": spaces.Box(0, 360, shape=(1,), dtype=int)
            }
        )

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert obstacle_type is None or obstacle_type in self.metadata["obstacle_types"]
        assert target_behavior is None or target_behavior in self.metadata["target_behaviors"]

        self.render_mode = render_mode

        self.current_round_reward = 0
        self.current_step_count = 0 

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": np.array(self._agent_location, dtype=int),
            "target": np.array(self._target_location, dtype=int),
            "agent_angle": np.array([self.agent_angle], dtype=int),
            "target_angle": np.array([self.target_angle], dtype=int)
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "obstacles": self._obstacles
        }

    def step(self, action):
        move_action = action["move"]
        view_angle_action = action["view_angle"]

        old_distance = np.linalg.norm(self._agent_location - self._target_location)
        self.agent_angle = view_angle_action
        new_location = self._agent_location.copy()

        state = self._get_obs()

        if move_action != 4:
            direction = self._action_to_direction[move_action]
            new_location = np.clip(self._agent_location + direction, 0, self.size - 1)
            if not self._is_collision(new_location):
                self._agent_location = new_location

        if self.target_behavior == 'random' or None:
            self.target_angle = (self.target_angle + np.random.randint(-45, 46)) % 360

        if self.target_behavior == 'circle':
            self.target_angle = (self.target_angle + 45) % 360

        agent_wins = self._check_visibility_agent()
        target_wins = self._check_visibility_target()
        terminated = agent_wins or target_wins 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        new_distance = np.linalg.norm(self._agent_location - self._target_location)
        reward = self._calculate_reward_variant3(old_distance, new_distance, agent_wins, target_wins, action, state)

        self.current_round_reward += reward 
        self.current_step_count += 1

        # Check for losing conditions
        terminated = agent_wins or target_wins or \
                     self.current_round_reward <= self.negative_reward_threshold or \
                     self.current_step_count >= self.max_steps_per_round

        print(f"Reward: {self.current_round_reward}")
        print(self.current_step_count)

        return observation, reward, terminated, False, info

    def _calculate_reward(self, old_distance, new_distance, agent_wins, target_wins):
        """
        Calculates the reward for the red agent.
        """
        reward = -0.1 # Small penalty for each step

        if agent_wins:
            reward += 100  # Large reward for catching the target
        elif target_wins:
            reward -= 100  # Large penalty for being caught

        distance_change = old_distance - new_distance
        reward += distance_change * 2  # Reward/penalty for getting closer/further

        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._place_obstacles()
        self.current_round_reward = 0
        self.current_step_count = 0 

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while self._is_collision(self._agent_location):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.agent_angle = self.np_random.integers(0, 360)

        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while np.array_equal(self._target_location, self._agent_location) or self._is_collision(self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.target_angle = self.np_random.integers(0, 360)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
    
    def _is_facing_target(self, agent_location, target_location, agent_angle):
        """
        Checks if the agent is facing the target within a certain tolerance.
        """
        delta = target_location - agent_location
        target_angle = (np.degrees(np.arctan2(delta[1], delta[0])) + 360) % 360 
        angle_difference = abs(target_angle - agent_angle) % 360

        # Check if the angle difference is within a tolerance (e.g., 45 degrees)
        return angle_difference <= 45  # You can adjust the tolerance

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Red and Blue')
            icon = pygame.image.load('logo_mini.png')
            pygame.display.set_icon(icon)
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        for obstacle in self._obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(obstacle),
                    (pix_square_size, pix_square_size),
                ),
            )

        self._draw_view_cone(canvas, self._target_location, self.target_angle, (0, 0, 255))

        self._draw_view_cone(canvas, self._agent_location, self.agent_angle, (255, 0, 0))

        # target
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # agent
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human" and self.window is not None:
            try:
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
            except pygame.error as e:
                print(f"Pygame error: {e}")
                self.close()

            self.clock.tick(self.metadata["render_fps"])
        #else:
            #return np.transpose(
            #    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            #)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _is_collision(self, location):
        return any(np.array_equal(location, obstacle) for obstacle in self._obstacles)

    def _calculate_num_obstacles(self):
        total_cells = self.size * self.size
        return int(total_cells * self.obstacle_percentage)

    def _place_obstacles(self):
        if self.obstacle_type == 'preset':
            self._obstacles = []

            num_cells = self._calculate_num_obstacles()

            while num_cells >= 0:
                row_index = np.random.randint(len(self.obstacle_presets))

                last_pair = self.obstacle_presets[row_index][-1]

                dx = np.random.randint(0, self.size - last_pair[0] + 1)
                dy = np.random.randint(0, self.size - last_pair[1] + 1)

                obstacle = [(pair[0] + dx, pair[1] + dy) for pair in self.obstacle_presets[row_index][:-1]]
                self._obstacles.extend(obstacle)
                num_cells = num_cells - len(obstacle[:-1])

            return

        if self.obstacle_type == 'random' or None:
            self._generate_obstacles()

    def _generate_obstacles(self):
        num_obstacles = np.random.randint(self.min_obstacles, self.max_obstacles + 1)
        self._obstacles = []
        for _ in range(num_obstacles):
            obstacle_width = np.random.randint(self.min_obstacles_len, self.max_obstacles_len + 1)
            obstacle_height = np.random.randint(self.min_obstacles_len, self.max_obstacles_len + 1)
            start_x = np.random.randint(0, self.size)
            start_y = np.random.randint(0, self.size)

            new_obstacles = [
                (start_x + dx, start_y + dy)
                for dx in range(obstacle_width)
                for dy in range(obstacle_height)
                if start_x + dx < self.size and start_y + dy < self.size
            ]

            self._obstacles.extend(new_obstacles)

    def _is_in_view(self, observer_location, target_location, angle):
        relative_pos = target_location - observer_location
        distance = np.linalg.norm(relative_pos)
        if distance > self.view_distance:
            return False

        angle_to_target = (np.degrees(np.arctan2(relative_pos[1], relative_pos[0])) - angle) % 360
        half_view_angle = self.view_angle / 2

        if angle_to_target > 180:
            angle_to_target -= 360

        in_view = -half_view_angle <= angle_to_target <= half_view_angle

        if in_view and not self._is_obstructed(observer_location, target_location):
            return True
        return False

    def _is_obstructed(self, observer_location, target_location):
        obs_line = _bresenham(observer_location, target_location)
        obstacles_set = set(self._obstacles)
        for point in obs_line:
            if point in obstacles_set:
                return True
        return False

    def _check_visibility_target(self):
        target_visible = self._is_in_view(self._target_location, self._agent_location, self.target_angle)
        return target_visible

    def _check_visibility_agent(self):
        agent_visible = self._is_in_view(self._agent_location, self._target_location, self.agent_angle)
        return agent_visible

    def _draw_view_cone(self, canvas, location, angle, color):
        pix_square_size = self.window_size / self.size
        agent_pos = (location[0] * pix_square_size + pix_square_size / 2,
                     location[1] * pix_square_size + pix_square_size / 2)

        alpha_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        alpha_color = (*color, 50)

        for d in range(1, self.view_distance + 1):
            for theta in np.linspace(-self.view_angle / 2, self.view_angle / 2, num=80):
                rad = math.radians(angle + theta)
                x = int(agent_pos[0] + d * pix_square_size * math.cos(rad))
                y = int(agent_pos[1] + d * pix_square_size * math.sin(rad))

                if 0 <= x < self.window_size and 0 <= y < self.window_size:
                    end_point = (x, y)
                    if self._is_obstructed(location, (x // pix_square_size, y // pix_square_size)):
                        break

                    pygame.draw.line(alpha_surface, alpha_color, agent_pos, end_point, 1)

        canvas.blit(alpha_surface, (0, 0))

    
    def _calculate_reward_variant1(self, old_distance, new_distance, terminated):
        """
        Reward function variant 1:
        - Large reward for terminating (catching the target).
        - Small reward for getting closer.
        - Small penalty for moving away. 
        """
        reward = 0
        if terminated:
            reward = 100  # Large reward for catching the target
        elif new_distance < old_distance:
            reward = 1  # Small reward for getting closer
        elif new_distance > old_distance:
            reward = -1  # Small penalty for moving away
        return reward

    def _calculate_reward_variant2(self, old_distance, new_distance, terminated):
        """
        Reward function variant 2:
        - Larger reward based on how much closer the agent got.
        - Larger penalty based on how much further the agent moved.
        """
        reward = 0
        if terminated:
            reward = 100 
        else:
            distance_change = old_distance - new_distance
            reward = distance_change * 2  # Scale the reward/penalty
        return reward

    def _calculate_reward_variant3(self, old_distance, new_distance, agent_wins, target_wins, action, state):
        """
        Reward function variant 3:
        - Encourages the agent to get closer to the target while facing it.
        - Penalizes collisions with obstacles.
        - Provides a bonus for being closer and looking at the target. 
        """
        reward = 0

        if agent_wins:
            reward += 100  # Large reward for catching the target
        elif target_wins:
            reward -= 100  # Large penalty for being caught
        else:
            # Reward for getting closer
            distance_change = old_distance - new_distance
            reward += distance_change * 5  # Увеличиваем значимость сближения

            # Penalize for not facing the target
            delta = self._target_location - self._agent_location
            desired_angle = (np.degrees(np.arctan2(delta[1], delta[0])) + 360) % 360
            angle_difference = abs(desired_angle - self.agent_angle) % 360

            reward -= (angle_difference / 360) * 5  # Усиливаем наказание за угол отклонения

            # Bonus for being close and facing the target
            if new_distance <= self.view_distance and angle_difference <= 45:
                reward += 5  # Увеличен бонус

            # Optionally return the penalty for collisions
            if self._is_collision(self._agent_location):
                reward -= 5  # Штраф за столкновения

        return reward
