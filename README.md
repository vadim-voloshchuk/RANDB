# Red and Blue Environment

This project implements a custom Gymnasium environment called "Red and Blue", simulating a simple game where a red agent tries to "catch" a blue target within its field of view while avoiding obstacles.

## Overview

The environment features a grid-based world where the red agent and blue target can move in discrete steps. The agent has a limited field of view, making the environment partially observable. Obstacles block the agent's movement and line of sight. The game ends when the agent "catches" the target (has it in its field of view) or when the agent loses (collides with an obstacle, target catches the agent, negative reward threshold reached, or maximum steps exceeded).


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/red-and-blue.git 
   cd red-and-blue
   ```

## Usage

### Running the Environment

You can run the environment with different agents and settings using the `main.py` script. Here's how to use it:

```bash
python main.py [arguments]
```

**Arguments:**

* `--agent`:  Agent to use (heuristic, neural, advanced-neural, rnn, qlearning). Default: "neural".
* `--size`: Size of the environment grid. Default: 100.
* `--fps`: Frames per second for rendering. Default: 10.
* `--obstacle_type`: Type of obstacles (random, preset). Default: "random".
* `--obstacle_percentage`: Percentage of obstacles. Default: 0.05.
* `--target_behavior`: Target behavior (circle, random). Default: "circle".
* `--episodes`: Number of episodes to run (for training agents). Default: 1000.
* `--load_q_table`: Path to load Q-table from (for qlearning agent).
* `--save_q_table`: Path to save Q-table to (for qlearning agent).
* `--output_dir`: Directory to save logs and results. Default: "output".

**Example:**

To run the environment with a neural agent on a 50x50 grid for 500 episodes:

```bash
python main.py --agent neural --size 50 --episodes 500
```

### Agents

The project includes several agents that can be used in the environment:

* **Heuristic Agent:** A simple agent that follows basic rules (e.g., move towards the target if visible).
* **Neural Agent:** A neural network-based agent that learns to play the game using reinforcement learning.
* **Advanced Neural Agent:** A more complex neural network agent (e.g., with convolutional layers for better feature extraction).
* **RNN Agent:** An agent using Recurrent Neural Networks (RNNs), specifically LSTMs, to handle the sequential nature of the environment.
* **Q-Learning Agent:** An agent that uses the Q-learning algorithm for reinforcement learning.


## Project Structure

...