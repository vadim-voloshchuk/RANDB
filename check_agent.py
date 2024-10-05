import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from demo_vis import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Инициализация среды
env = gym.make("RedAndBlue-v0.1", render_mode=None, size=50, target_behavior='circle')

# Агент DQN (как ранее)
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space["agent"].shape[0] + 2  # Угол обзора и цель
        self.action_dim = env.action_space["move"].n

        # Загрузка модели
        self.q_network = DQN(self.state_dim, self.action_dim).to(device)
        self.q_network.load_state_dict(torch.load("dqn_weights_episode_500.pth"))  # Укажите нужный файл с весами
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
            predicted_angle = angle_values.argmax().item()  # Получаем индекс угла
        return action, predicted_angle

    def evaluate(self, num_episodes=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, predicted_angle = self.select_action(self._flatten_state(state))
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

    def _flatten_state(self, state):
        agent = state["agent"]
        agent_angle = state["agent_angle"]
        target_angle = state["target_angle"]
        return np.concatenate([agent, agent_angle, target_angle])

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

# Инициализация агента и поток для графика
agent = DQNAgent(env)

# Создание графика
plt.figure(figsize=(10, 5))
ani = FuncAnimation(plt.gcf(), update_plot, interval=100)  # Обновление каждые 100 мс

# Запуск оценки в отдельном потоке
evaluation_thread = threading.Thread(target=agent.evaluate, args=(100,))
evaluation_thread.start()

plt.show()  # Показ графика

# Ожидание завершения потока оценки
evaluation_thread.join()

env.close()
