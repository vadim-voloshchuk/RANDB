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
        elif args.agent == "pretentious_pytorch":
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
    try:
        print("Initializing PretentiousAgent...")
        agent = PretentiousAgent()
        print("PretentiousAgent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize PretentiousAgent: {e}")
