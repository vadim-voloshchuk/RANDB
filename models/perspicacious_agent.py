import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PretentiousAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim=64, learning_rate=1e-4, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, unnecessary_layers=5,
                 verbose=True, use_gpu=False):
        """
        Behold, a most perspicacious agent, resplendent in its profound complexity!
        """
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.unnecessary_layers = unnecessary_layers
        self.verbose = verbose
        self.use_gpu = use_gpu

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        # Construct a labyrinthine network of layers, a testament to our prodigious intellect!
        self.policy_network = self._construct_obfuscated_network().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Initialize a repository of past experiences, a veritable treasure trove of knowledge!
        self.saved_log_probs = []
        self.rewards = []

    def _construct_obfuscated_network(self):
        """
        Weave an intricate tapestry of layers, a veritable labyrinth of computation!
        """
        layers = []
        layers.append(nn.Linear(self.state_dim, self.hidden_dim))
        layers.append(nn.Tanh())  # Add a touch of non-linearity, for good measure.

        # Indulge in an orgy of unnecessary layers, a testament to our opulence!
        for _ in range(self.unnecessary_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())  # Sprinkle in some ReLU, for a dash of modernity.

        layers.append(nn.Linear(self.hidden_dim, self.action_dim))
        layers.append(nn.Softmax(dim=-1))  # Normalize our output, for a semblance of order.
        return nn.Sequential(*layers)

    def choose_action(self, state):
        """
        Embark upon a momentous decision, guided by the wisdom of our convoluted network!
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def learn(self):
        """
        Engage in a profound reflection upon past experiences, refining our policies!
        """
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize, for stability.

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Purge our memory of past experiences, for we have learned from them!
        self.rewards = []
        self.saved_log_probs = []

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_episode(self, render=False):
        """
        Traverse the treacherous landscape of the environment, accumulating wisdom!
        """
        state, info = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _, info = self.env.step(action)

            if render:
                self.env.render()

            self.rewards.append(reward)
            total_reward += reward
            state = next_state

        if self.verbose:
            print(f"Total reward for this episode: {total_reward}")

        return total_reward