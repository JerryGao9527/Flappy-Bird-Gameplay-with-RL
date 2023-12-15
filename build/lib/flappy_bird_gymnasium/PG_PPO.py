import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import pandas as pd

from model import REINFORCE, ValueNetwork  

class PPO:
    def __init__(self, policy_class, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = policy_class().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = policy_class().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self, memory):
        # Convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = old_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, old_rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

def train_ppo(env, policy, value_network, policy_optimizer, value_optimizer, epochs, batch_size, gamma, clip_param, value_loss_coef, entropy_coef):
    for epoch in range(epochs):
        episode_durations = []
        all_advantages = []
        all_returns = []
        all_log_probs = []

        # Collect trajectories
        for _ in range(batch_size):
            states, actions, rewards, values, log_probs = [], [], [], [], []
            full_state = env.reset()
            state = full_state[0]  # Extract the state array
            duration = 0
            while True:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action_probs, value = policy(state_tensor), value_network(state_tensor)

                action = np.random.choice(np.array([0, 1]), p=action_probs.detach().cpu().numpy()[0])
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(torch.log(action_probs.squeeze(0)[action] + 1e-9))

                duration += 1
                if done:
                    episode_durations.append(duration)
                    break
                state = next_state

            # Compute returns and advantages
            returns, advantages = [], []
            G, A = 0, 0
            for step in reversed(range(len(rewards))):
                G = rewards[step] + gamma * G
                delta = rewards[step] + gamma * values[step + 1] - values[step] if step < len(rewards) - 1 else rewards[step] - values[step]
                A = delta + gamma * A
                returns.insert(0, G)
                advantages.insert(0, A)

            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_log_probs.extend(log_probs)

        # Normalize advantages
        all_advantages = torch.tensor(all_advantages).to(device)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Convert lists to tensors
        all_states = torch.tensor(states, dtype=torch.float32).to(device)
        all_actions = torch.tensor(actions).to(device)
        all_returns = torch.tensor(all_returns, dtype=torch.float32).to(device)
        all_old_log_probs = torch.tensor(all_log_probs).to(device)

        # Update the policy and value network
        for _ in range(4):  # 4 is the number of optimization epochs
            log_probs, state_values, dist_entropy = policy.evaluate(all_states, all_actions)

            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - all_old_log_probs)

            # Compute clipped objective
            surr1 = ratios * all_advantages
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * all_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = F.mse_loss(state_values.squeeze(), all_returns)

            # Total loss
            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * dist_entropy.mean()

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            policy_optimizer.step()
            value_optimizer.step()

        # Log average episode duration
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Average Duration {np.mean(episode_durations)}")

    return policy, value_network


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--update_timestep', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--K_epochs', type=int, default=4)
    args = parser.parse_args()

    # Environment Setup
    env = gymnasium.make("FlappyBird-v0", render_mode = None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Network Parameters
    input_size = env.observation_space.shape[0]
    output_size = 2

    # Create Policy Network and Optimizer
    policy = REINFORCE(input_size, output_size).to(device)
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