import cProfile
import pstats
import io
import argparse
import logging
import json
import os

import gymnasium as gym
import redandblue

from models.simple_heuristic import HeuristicAgent
from models.neural import NeuralAgent
from models.advanced_neural import AdvancedNeuralAgent
from models.rnn_agent import RNNNeuralAgent

#from models.q_learning import QLearningAgent
#import models.q_learning

import numpy as np

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

    # Create the environment based on command-line arguments
    env_show = gym.make(
        'RedAndBlue-v0.1', 
        render_mode='human', 
        size=args.size, 
        fps=args.fps,
        obstacle_type=args.obstacle_type, 
        obstacle_percentage=args.obstacle_percentage, 
        target_behavior=args.target_behavior
    )

    # Choose the agent based on command-line arguments
    if args.agent == "heuristic":
        agent = HeuristicAgent() 
    elif args.agent == "neural":
        agent = NeuralAgent()
    elif args.agent == "advanced-neural":
        agent = AdvancedNeuralAgent()
    elif args.agent == "rnn":
        agent = RNNNeuralAgent()
    #elif args.agent == "qlearning":
    #    agent = QLearningAgent(env_show)
    #    if args.load_q_table:
    #        agent.load_q_table(args.load_q_table)
    #    agent.train(episodes=args.episodes)
    #    if args.save_q_table:
    #        agent.save_q_table(args.save_q_table)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    # Run the chosen agent
    agent.run_model(env_show, episodes=args.episodes, output_dir=args.output_dir) 

    logging.info("Finished running Red and Blue environment")

    env_show.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red and Blue environment runner with profiling.")
    parser.add_argument("--agent", type=str, default="neural", choices=["heuristic", "neural", "advanced-neural","rnn"], help="Agent to use (heuristic, neural, advanced-neural, rnn)")
    parser.add_argument("--size", type=int, default=100, help="Size of the environment grid")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for rendering")
    parser.add_argument("--obstacle_type", type=str, default="random", choices=["random", "preset"], help="Type of obstacles")
    parser.add_argument("--obstacle_percentage", type=float, default=0.05, help="Percentage of obstacles")
    parser.add_argument("--target_behavior", type=str, default="circle", choices=["circle", "random"], help="Target behavior")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run (for training agents)")
    #parser.add_argument("--load_q_table", type=str, help="Path to load Q-table from (for qlearning agent)")
    #parser.add_argument("--save_q_table", type=str, help="Path to save Q-table to (for qlearning agent)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save logs and results")

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