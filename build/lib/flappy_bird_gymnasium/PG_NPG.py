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

import torch

def compute_fisher_information(policy, trajectories_states, trajectories_actions, log_probs_grads, lambda_reg):
    """
    Compute the Fisher Information Matrix for the given policy using pre-computed log probabilities and gradients.

    :param policy: The policy network.
    :param trajectories_states: List of trajectories, where each trajectory is a list of states.
    :param trajectories_actions: List of trajectories, where each trajectory is a list of actions.
    :param log_probs_grads: List of pre-computed gradients of log probabilities for each trajectory.
    :param lambda_reg: Regularization constant used to force positive-definiteness.
    :return: Fisher Information Matrix.
    """
    policy.eval()  # Set the network to evaluation mode
    fisher_information = torch.zeros((policy.num_parameters(), policy.num_parameters()))
    N = len(trajectories_states)  # Number of trajectories

    for idx, (states, actions) in enumerate(zip(trajectories_states, trajectories_actions)):
        trajectory_fim = torch.zeros_like(fisher_information)
        H = len(states)  # Length of the trajectory

        for h in range(len(states)):
            grads = log_probs_grads[idx][h]

            # Accumulate FIM for the trajectory
            trajectory_fim += torch.outer(grads, grads)

        trajectory_fim /= H
        fisher_information += trajectory_fim

    fisher_information /= N
    fisher_information += lambda_reg * torch.eye(fisher_information.size(0))

    return fisher_information

def compute_eta(policy_gradient, fisher_matrix_inv, delta):
    """
    Compute the learning rate (eta) based on the trust region constraint.

    :param policy_gradient: Computed policy gradient.
    :param fisher_matrix_inv: Inverse of the Fisher Information Matrix.
    :param delta: Trust region size.
    :return: Learning rate (eta).
    """
    sqrt_term = torch.sqrt(policy_gradient.t() @ fisher_matrix_inv @ policy_gradient) + 1e-6
    return torch.sqrt(delta / sqrt_term)


def npg(env, policy, optimizer, epochs, batch_size, gamma, delta, lambda_reg, epsilon):
    """
    Train a policy using the Natural Policy Gradient algorithm with mini-batches and epsilon decay.

    :param env: The game environment.
    :param policy: The policy model to be trained.
    :param optimizer: The optimizer used for training.
    :param epochs: Number of epochs to train the model.
    :param batch_size: Number of trajectories for each mini-batch.
    :param gamma: Discount factor for future rewards.
    :param delta: Trust region size.
    :param lambda_reg: Regularization constant for FIM.
    :param epsilon: The starting value for epsilon in the epsilon-greedy strategy.
    :return: List of episode durations for each epoch.
    """
    episode_durations = []

    for epoch in range(epochs):
        trajectories_states = []
        trajectories_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_log_probs_grads = []

        # Collect trajectories
        for _ in range(batch_size):
            trajectory_states = []
            trajectory_actions = []
            log_probs = []
            rewards = []
            log_probs_grads_per_trajectory = []

            full_state = env.reset()
            state = full_state[0]  # Extract the state array
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
                trajectory_states.append(state)
                trajectory_actions.append(action)
                if reward == 0.1:
                    reward *= max((100 - t) / 100, 0)
                log_prob = torch.log(action_probs.squeeze(0)[action] + 1)# For numerical stability 
                # log_prob = torch.log(action_probs.squeeze(0)[action].clamp(min=1e-9))

                log_prob.backward()
                log_probs.append(log_prob) 
                grads = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
                log_probs_grads_per_trajectory.append(grads.detach())
                rewards.append(reward)
                
                if done:
                    break

                state = next_state

            trajectories_states.append(trajectory_states)
            trajectories_actions.append(trajectory_actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)
            episode_durations.append(len(rewards))
            batch_log_probs_grads.append(log_probs_grads_per_trajectory)
        
        # Compute Fisher Information Matrix
        fisher_matrix = compute_fisher_information(policy, trajectories_states, trajectories_actions, batch_log_probs_grads, lambda_reg)
        fisher_matrix_inv = torch.inverse(fisher_matrix)

        # Compute Policy Gradient
        batch_policy_gradient = []
        for trajectory_log_probs_grads, trajectory_rewards in zip(batch_log_probs_grads, batch_rewards):
            policy_gradient = 0
            R = 0
            for t in reversed(range(len(trajectory_rewards))):
                R = trajectory_rewards[t] + gamma * R
                policy_gradient += trajectory_log_probs_grads[t] * R
            batch_policy_gradient.append(policy_gradient)

        # Update Policy
        optimizer.zero_grad()
        flat_policy_gradient = torch.mean(torch.stack(batch_policy_gradient), 0)
        eta = compute_eta(flat_policy_gradient, fisher_matrix_inv, delta)
        update = eta * fisher_matrix_inv @ flat_policy_gradient

        if torch.isnan(update).any():
            print("Warning: NaN values detected in the update tensor.")

        # Apply update to policy parameters
        idx = 0
        for param in policy.parameters():
            numel = param.numel()
            param.data.copy_(param.data + update[idx:idx + numel].reshape(param.shape))
            idx += numel

        # Epsilon Decay
        epsilon = max(0.9995 * epsilon, 0.1)

        # Print average duration every 1000 episodes
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Average Duration {np.mean(episode_durations[-1000:])}")

    return episode_durations


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--delta', type=float, default=0.01, help='Trust region size for NPG')
    parser.add_argument('--lambda_reg', type=float, default=1e-2, help='Regularization constant for FIM')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Starting epsilon for the epsilon-greedy strategy')
    args = parser.parse_args()

    # Environment Setup
    env = gymnasium.make("FlappyBird-v0", render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Network Parameters
    input_size = env.observation_space.shape[0]
    hidden_size = 128
    output_size = 2

    # Create Policy Network
    policy = REINFORCE(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training using NPG
    durations = npg(env, policy, optimizer, args.num_episodes, args.batch_size, args.gamma, args.delta, args.lambda_reg, args.epsilon)
    durations_series = pd.Series(durations)

    # Plotting and saving results
    window_size = 50
    rolling_mean = durations_series.rolling(window=window_size).mean()

    plt.figure(figsize=(15, 8))
    plt.plot(durations, label='Individual Game Duration', color='blue', alpha=0.4)
    plt.plot(rolling_mean, label=f'{window_size}-Game Rolling Mean Duration', color='orange', linewidth=2)
    plt.title("Survival Time vs Number of Games Played")
    plt.xlabel("Games Played")
    plt.ylabel("Survival Time")
    plt.legend()
    plt.grid(True)
    plt.savefig('flappy_bird_gymnasium/figs/Duration_vs_Epochs.png')
    plt.close()

    # Save Model Weights
    torch.save(policy.state_dict(), 'flappy_bird_gymnasium/weights/PG_NPG.pth')

    env.close()