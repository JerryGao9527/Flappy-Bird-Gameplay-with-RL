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
        i = 0

        while True:
            i += 1
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():  # Disable gradient calculation
                action_probs = policy(state)
                # print(action_probs.numpy())
            # action = np.argmax(action_probs.numpy())  # Choose action with highest probability
            action = np.random.choice(np.array([0, 1]), p=action_probs.detach().numpy()[0])
            state, reward, done, _, info = env.step(action)
            print(state)
            total_reward += reward

            if done:
                break


# Test Environment
env = gymnasium.make("FlappyBird-v0", render_mode="human")

# Load Model
input_size = env.observation_space.shape[0]
hidden_size = 128
output_size = 2
policy = REINFORCE(input_size, hidden_size, output_size)
# path = 'flappy_bird_gymnasium/weights/PG_REINFORCE.pth'
path = 'flappy_bird_gymnasium/weights/PG_NPG.pth'
policy.load_state_dict(torch.load(path))

# Test the Policy
test_episodes = 1
test_policy(env, policy, test_episodes)

env.close()
