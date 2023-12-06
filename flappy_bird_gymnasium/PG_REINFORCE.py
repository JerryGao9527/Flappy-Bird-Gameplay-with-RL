import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import pandas as pd

from model import REINFORCE

def reinforce(env, use_baseline, policy, optimizer, epochs, batch_size, gamma, epsilon):
    """
    Train a policy using the REINFORCE algorithm with mini-batches and epsilon decay.

    :param env: The game environment.
    :param policy: The policy model to be trained.
    :param optimizer: The optimizer used for training.
    :param epochs: Number of epochs to train the model.
    :param batch_size: Number of trajectories for each mini-batch.
    :param gamma: Discount factor for future rewards.
    :param epsilon: The starting value for epsilon in the epsilon-greedy strategy.
    :return: List of episode durations for each epoch.
    """
    episode_durations = []

    for epoch in range(epochs):
        batch_log_probs = []
        batch_rewards = []
        for _ in range(batch_size):
            full_state = env.reset()
            state = full_state[0]  # Extract the state array
            log_probs = []
            rewards = []

            # Generate a trajectory
            for t in range(10000):      
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action_probs = policy(state_tensor)
                # Exploration strategy: Epsilon-greedy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    action = np.random.choice(np.array([0, 1]), p=action_probs.detach().numpy()[0]) # Greedy action

                next_state, reward, done, _, info = env.step(action)
                log_probs.append(torch.log(action_probs.squeeze(0)[action] + 1)) # For numerical stability 
                rewards.append(reward)
                
                if done:
                    break

                state = next_state

            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)

        # Calculate batch losses below
        batch_policy_losses = []
        if use_baseline:
            b = np.mean([np.mean(traj) for traj in batch_rewards])

        for trajectory in range(batch_size):
            trajectory_log_probs = batch_log_probs[trajectory]
            trajectory_rewards = batch_rewards[trajectory]
            policy_loss = 0
            R = 0
            for t in reversed(range(len(trajectory_rewards))):
                R = trajectory_rewards[t] + gamma * R
                if use_baseline:
                    # Calculate policy gradient with the use of a constant baseline
                    policy_loss -= trajectory_log_probs[t] * (R - b)
                else:
                    policy_loss -= trajectory_log_probs[t] * R

            batch_policy_losses.append(policy_loss)
            episode_durations.append(len(trajectory_rewards))

        # Update policy
        epoch_policy_loss = torch.stack(batch_policy_losses).mean()
        optimizer.zero_grad()
        epoch_policy_loss.backward()
        optimizer.step()

        epsilon = max(0.05, max(0.9995 * epsilon, 0.1))

        # Print average duration every 1000 episodes
        if epoch % 1000 == 0:
            print(f"Played {epoch} episodes with average duration {np.mean(episode_durations[-1000:])}")

    return episode_durations


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_baseline', type=bool, default=True, help='whether to use baseline method')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Starting epsilon for the epsilon-greedy strategy')
    args = parser.parse_args()

    # Environment Setup
    env = gymnasium.make("FlappyBird-v0", render_mode = None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Network Parameters
    input_size = env.observation_space.shape[0]
    hidden_size = 128
    output_size = 2

    # Create Policy Network and Optimizer
    policy = REINFORCE(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    # Training
    durations = reinforce(env, args.use_baseline, policy, optimizer, args.num_episodes, args.batch_size, args.gamma, args.epsilon)
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
    # plt.yscale('log')  
    plt.legend()
    plt.grid(True)
    plt.savefig('flappy_bird_gymnasium/figs/Duration_vs_Epochs.png')
    plt.close()

    # Save Model Weights
    torch.save(policy.state_dict(), 'flappy_bird_gymnasium/weights/PG_REINFORCE.pth')

    env.close()