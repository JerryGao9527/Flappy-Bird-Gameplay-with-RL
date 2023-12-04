import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import REINFORCE
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import pandas as pd

def reinforce(env, policy, optimizer, epochs, batch_size, gamma):
    episode_durations = []
    epsilon = 0.5

    for epoch in range(epochs):
        epoch_policy_losses = []
        for trajectory in range(batch_size):
            full_state = env.reset()
            state = full_state[0]  # Extract the state array
            rewards = []
            log_probs = []
            steps = 0

            # Generate a trajectory
            for t in range(10000):     
                steps += 1     
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action_probs = policy(state_tensor)
                # print(action_probs)
                # Exploration strategy: Epsilon-greedy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    action = np.random.choice(np.array([0, 1]), p=action_probs.detach().numpy()[0])
                # print(action)
                next_state, reward, done, _, info = env.step(action)
                rewards.append(reward)
                log_probs.append(torch.log(action_probs.squeeze(0)[action] + 1))

                if done:
                    break

                state = next_state

            # Calculate policy gradient
            policy_loss = 0
            R = 0
            for t in reversed(range(len(rewards))):
                R = rewards[t] + gamma * R
                policy_loss -= log_probs[t] * R

            epoch_policy_losses.append(policy_loss)
            
            episode_durations.append(steps)

        # Update policy
        epoch_policy_loss = torch.stack(epoch_policy_losses).mean()
        optimizer.zero_grad()
        epoch_policy_loss.backward()
        optimizer.step()

        epsilon = max(0.05, 0.999 * epsilon)

        if epoch % 1000 == 0:
            print(f"Played {epoch} games with average duration {np.mean(episode_durations)}")

    return episode_durations

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
args = parser.parse_args()

# Environment Setup
env = gymnasium.make("FlappyBird-v0", render_mode = None)

# Network Parameters
input_size = env.observation_space.shape[0]
hidden_size = 128
output_size = 2

# Create Policy Network and Optimizer
policy = REINFORCE(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)

# Training
durations = reinforce(env, policy, optimizer, args.num_episodes, args.batch_size, args.gamma)
durations_series = pd.Series(durations)

# Calculate the rolling mean with a window size of your choice
# Adjust the window size to smooth the trend line
window_size = 50
rolling_mean = durations_series.rolling(window=window_size).mean()

plt.figure(figsize=(15, 8))
plt.plot(durations, label='Individual Game Duration', color='blue', alpha=0.4)
plt.plot(rolling_mean, label=f'{window_size}-Game Rolling Mean Duration', color='orange', linewidth=2)
plt.title("Survival Time vs Number of Games Played")
plt.xlabel("Games Played")
plt.ylabel("Survival Time")
# plt.yscale('log')  # Set the y-axis to a logarithmic scale if needed
plt.legend()
plt.grid(True)
plt.savefig('flappy_bird_gymnasium/figs/Duration_vs_Epochs.png')
plt.close()

# Save Model Weights
torch.save(policy.state_dict(), 'flappy_bird_gymnasium/weights/PG_REINFORCE.pth')

env.close()