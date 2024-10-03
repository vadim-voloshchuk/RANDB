import pandas as pd
import matplotlib.pyplot as plt

# Чтение файла CSV
data = pd.read_csv('training_history.csv')

# Вывод первых нескольких строк данных
print(data.head())

# Отображение графика награды по эпизодам
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Reward'], label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward over Episodes')
plt.legend()
plt.grid(True)
plt.show()

# Отображение выигрышей и проигрышей
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Wins'], label='Wins')
plt.plot(data['Episode'], data['Losses'], label='Losses')
plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('Wins and Losses over Episodes')
plt.legend()
plt.grid(True)
plt.show()

# Можно добавить дополнительные графики и анализ
