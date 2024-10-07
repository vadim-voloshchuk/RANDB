import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('TkAgg')  # Убедитесь, что используете правильный бэкенд
import matplotlib.pyplot as plt
import threading
import redandblue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация среды
env = gym.make("RedAndBlue-v0.1", render_mode="human", size=50, target_behavior='circle')

# Сеть для DQN с поддержкой AMP
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, angle_dim=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Увеличено количество нейронов
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Третий слой увеличен
        self.action_head = nn.Linear(128, output_dim)
        self.angle_head = nn.Linear(128, angle_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # Применяем активацию для третьего слоя
        action = self.action_head(x)
        angle = torch.sigmoid(self.angle_head(x)) * 360  # Масштабирование угла до диапазона [0, 360]
        return action, angle

    
# Агент DQN
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.state_dim = env.observation_space["agent"].shape[0] + env.observation_space["target"].shape[0] + 2 + 1  + self.size * self.size# Угол обзора и цель
        self.action_dim = env.action_space["move"].n

        # Загрузка модели
        self.q_network = DQN(self.state_dim, self.action_dim).to(device)
        self.q_network.load_state_dict(torch.load("dqn_weights_episode_1000.pth"), strict=False)
        self.q_network.eval()

        # Статистика
        self.total_reward = 0
        self.episode_wins = 0
        self.episode_losses = 0
        self.rewards_history = []
        self.wins_history = []
        self.losses_history = []
        self.lock = threading.Lock()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values, angle_values = self.q_network(state)
            action = action_values.argmax().item()
            predicted_angle = angle_values.item()  # Получаем индекс угла
        return action, predicted_angle

    def evaluate(self, num_episodes=100):
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, predicted_angle = self.select_action(self._flatten_state(state, info))
                next_state, reward, done, _, _ = self.env.step({"move": action, "view_angle": predicted_angle})

                episode_reward += reward
                state = next_state
            
            self.total_reward += episode_reward

            # Обновление счёта выигрыш/проигрыш
            with self.lock:
                if episode_reward == 100:
                    self.episode_wins += 1
                elif episode_reward == -100:
                    self.episode_losses += 1
                
                self.rewards_history.append(episode_reward)
                self.wins_history.append(self.episode_wins)
                self.losses_history.append(self.episode_losses)

            # Вывод информации об эпизоде
            print(f"Episode {episode}: Reward {episode_reward}")

    def _flatten_state(self, state, info):
        agent = state["agent"]
        target = state["target"]
        agent_angle = state["agent_angle"]
        target_angle = state["target_angle"]
        distance = info["distance"]
        obs = info["obstacles"]
        
        # Инициализируем матрицу 100x100 (например, с препятствиями)
        grid = np.zeros((self.size, self.size))
        for obstacle in obs:
            grid[obstacle[0], obstacle[1]] = 1

        flattened_grid = grid.flatten()

        # Добавляем матрицу к остальным частям состояния
        return np.concatenate([agent, agent_angle, target, target_angle, [distance], flattened_grid])


# Функция для обновления графика
def update_plot(frame):
    with agent.lock:
        plt.clf()  # Очистка предыдущего графика
        plt.title('Wins and Losses Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Count')
        plt.plot(agent.wins_history, label='Wins', color='green')
        plt.plot(agent.losses_history, label='Losses', color='red')
        plt.legend()
        plt.xlim(0, len(agent.wins_history) + 10)  # Устанавливаем пределы по оси X
        plt.ylim(0, max(max(agent.wins_history), max(agent.losses_history), 10))  # Устанавливаем пределы по оси Y

# Инициализация агента
agent = DQNAgent(env)

# Создание графика
plt.figure(figsize=(10, 5))
plt.ion()  # Включение интерактивного режима

# Запуск оценки в отдельном потоке
evaluation_thread = threading.Thread(target=agent.evaluate, args=(100,))
evaluation_thread.start()

while evaluation_thread.is_alive():
    update_plot(None)  # Обновление графика
    plt.pause(0.1)  # Пауза для обновления графика

# Показ графика после завершения
plt.show()

# Ожидание завершения потока оценки
evaluation_thread.join()

env.close()
