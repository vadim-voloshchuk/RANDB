import cProfile
import pstats
import io
import argparse
import logging
import json
import os

# Импортируем TensorFlow для настройки GPU
import tensorflow as tf

import gymnasium as gym
import redandblue

from models.simple_heuristic import HeuristicAgent
from models.neural import NeuralAgent
from models.advanced_neural import AdvancedNeuralAgent
from models.rnn_agent import RNNNeuralAgent
from models.cnn_rnn import CNNRNNNeuralAgent
from models.qagent import QLearningAgent 
from models.tree_agent import DecisionTreeAgent
from models.perspicacious_agent import PretentiousAgent  # Добавьте импорт PretentiousAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configure log file
    log_file = os.path.join(args.output_dir, "run.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("Starting Red and Blue environment runner")
    logging.info(f"Arguments: {args}")

    # Настройка TensorFlow для использования GPU, если доступен
    if args.gpu:
        # Получаем список доступных GPU
        gpus = tf.config.list_physical_devices('GPU')
        logging.info(f"Available GPUs: {gpus}")

        if gpus:
            try:
                # Разрешаем использование первой доступной GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                logging.info(f"Using GPU: {logical_gpus}")

                # Дополнительные оптимизации для H100:
                # 1. Включаем XLA для компиляции графов TensorFlow
                tf.config.optimizer.set_jit(True)

                # 2. Устанавливаем смешанную точность (FP16)
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)

            except RuntimeError as e:
                # Ошибка настройки GPU, используем CPU
                logging.warning(f"Error setting GPU: {e}. Using CPU.")
        else:
            logging.warning("No GPU found. Using CPU.")
    else:
        logging.info("GPU usage not requested. Using CPU.")

    # Определяем режим рендеринга на основе аргумента --render
    render_mode = 'human' if args.render else None
    logging.info(f"Render mode set to: {render_mode}")

    # Создаем среду на основе аргументов командной строки
    try:
        env_show = gym.make(
            'RedAndBlue-v0.1',
            render_mode=render_mode,  # Используем выбранный режим рендеринга
            size=args.size,
            fps=args.fps,
            obstacle_type=args.obstacle_type,
            obstacle_percentage=args.obstacle_percentage,
            target_behavior=args.target_behavior
        )
    except Exception as e:
        logging.error(f"Failed to create environment: {e}")
        return

    # Выбираем агента на основе аргументов командной строки
    try:
        if args.agent == "heuristic":
            agent = HeuristicAgent() 
        elif args.agent == "neural":
            agent = NeuralAgent()
        elif args.agent == "advanced-neural":
            agent = AdvancedNeuralAgent()
        elif args.agent == "rnn":
            agent = RNNNeuralAgent()
        elif args.agent == "cnnrnn":
            agent = CNNRNNNeuralAgent()
        elif args.agent == "qlearning":
            agent = QLearningAgent()
        elif args.agent == "decision_tree":
            agent = DecisionTreeAgent()
        elif args.agent == "pretentious_pytorch":  # Добавьте выбор PretentiousAgent
            agent = PretentiousAgent()
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
    except Exception as e:
        logging.error(f"Failed to create agent: {e}")
        env_show.close()
        return
    
    logging.info("Agent initialized.")
    try:
        agent.run_model(env_show, episodes=args.episodes, output_dir=args.output_dir)
    except Exception as e:
        logging.error(f"Failed during agent execution: {e}")
    finally:
        logging.info("Finished running Red and Blue environment")
        env_show.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red and Blue environment runner with profiling.")
    parser.add_argument("--size", type=int, default=100, help="Size of the environment grid")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for rendering")
    parser.add_argument("--obstacle_type", type=str, default="random", choices=["random", "preset"], help="Type of obstacles")
    parser.add_argument("--obstacle_percentage", type=float, default=0.05, help="Percentage of obstacles")
    parser.add_argument("--target_behavior", type=str, default="circle", choices=["circle", "random"], help="Target behavior")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run (for training agents)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save logs and results")
    parser.add_argument("--agent", type=str, default="cnnrnn", 
                        choices=["heuristic", "neural", "advanced-neural", "rnn", "cnnrnn", 
                                 "qlearning", "decision_tree", "pretentious_pytorch"],  # Добавьте pretentious_pytorch
                        help="Agent to use") 

    # Добавляем аргументы для использования GPU и рендеринга
    parser.add_argument("--gpu", action="store_true", help="Use GPU for acceleration (if available)")
    parser.add_argument("--render", action="store_true", help="Enable visualization (rendering)")

    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()

    try:
        main(args)
    finally:
        pr.disable()

        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        profile_file = os.path.join(args.output_dir, "profile_stats.txt")
        with open(profile_file, "w") as f:
            f.write(s.getvalue())
        logging.info(f"Profiling results saved to: {profile_file}")
