import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import REINFORCE
import flappy_bird_gymnasium
import gymnasium

def test_policy(env, policy, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        total_reward = 0

        while True:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():  # Disable gradient calculation
                action_probs = policy(state)
                print(action_probs.numpy())
            action = np.argmax(action_probs.numpy())  # Choose action with highest probability
            state, reward, done, _, info = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


# Test Environment
env = gymnasium.make("FlappyBird-v0", render_mode="human")

# Load Model
input_size = env.observation_space.shape[0]
hidden_size = 128
output_size = 2
policy = REINFORCE(input_size, hidden_size, output_size)
policy.load_state_dict(torch.load('flappy_bird_gymnasium/weights/PG_REINFORCE.pth'))

# Test the Policy
test_episodes = 1
test_policy(env, policy, test_episodes)

env.close()
