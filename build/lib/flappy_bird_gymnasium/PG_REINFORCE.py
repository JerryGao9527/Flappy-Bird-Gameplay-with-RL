import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import REINFORCE
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt

def reinforce(env, policy, optimizer, epochs, batch_size, gamma):
    episode_durations = []
    for _ in range(epochs):
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
                epsilon = 0.2
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    action = np.random.choice(np.array([0, 1]), p=action_probs.detach().numpy()[0])
                # print(action)
                next_state, reward, done, _, info = env.step(action)
                print(reward)
                # if reward == 1:
                #     reward = 2
                # if reward == 0.1:
                #     reward = 0.01
                rewards.append(reward)
                log_probs.append(torch.log(action_probs.squeeze(0)[action]))

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


    return episode_durations

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
args = parser.parse_args()

# Environment Setup
env = gymnasium.make("FlappyBird-v0", render_mode = None)

# Network Parameters
input_size = env.observation_space.shape[0]
hidden_size = 128
output_size = 2

# Create Policy Network and Optimizer
policy = REINFORCE(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

# Training
durations = reinforce(env, policy, optimizer, args.num_episodes, args.batch_size, args.gamma)

# Plotting and saving the loss plot
plt.figure(figsize=(10, 6))
plt.plot(durations)
plt.title("Episode Duration per Epoch")
plt.xlabel("Number of games played")
plt.ylabel("Duration (Number of Steps)")
plt.grid(True)
plt.savefig('flappy_bird_gymnasium/figs/Duration_vs_Epochs.png')
plt.close()

# Save Model Weights
torch.save(policy.state_dict(), 'flappy_bird_gymnasium/weights/PG_REINFORCE.pth')

env.close()