import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium

from model import REINFORCE  # Assuming this is your policy network class

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--update_timestep', type=int, default=2000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--K_epochs', type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_ppo(env, policy, value_network, policy_optimizer, value_optimizer, epochs, batch_size, gamma, epsilon, clip_param, value_loss_coef, entropy_coef):
    for epoch in range(epochs):
        # Collect trajectories
        # (Similar to your npg function, but also store values and compute advantages)
        trajectories_states, trajectories_actions, trajectories_values, trajectories_advantages = [], [], [], []
        for _ in range(batch_size):
            states, actions, rewards, values, log_probs = [], [], [], [], []
            state = env.reset()[0]
            for t in range(10000):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action_probs = policy(state_tensor)
                value = value_network(state_tensor)

                action = np.random.choice(np.array([0, 1]), p=action_probs.detach().numpy()[0])
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(torch.log(action_probs.squeeze(0)[action] + 1e-9))

                if done:
                    break
                state = next_state

            # Compute returns and advantages
            returns, advantages = [], []
            G, A = 0, 0
            for step in reversed(range(len(rewards))):
                G = rewards[step] + gamma * G
                delta = rewards[step] + gamma * values[step + 1] - values[step] if step < len(rewards) - 1 else rewards[step] - values[step]
                A = delta + gamma * epsilon * A
                returns.insert(0, G)
                advantages.insert(0, A)

            trajectories_states.extend(states)
            trajectories_actions.extend(actions)
            trajectories_values.extend(values)
            trajectories_advantages.extend(advantages)

        # Update the policy and value network
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        policy_loss, value_loss = 0, 0
        for state, action, old_log_prob, return_, advantage in zip(trajectories_states, trajectories_actions, trajectories_log_probs, trajectories_returns, trajectories_advantages):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_tensor = torch.tensor([action])

            # Policy loss
            new_action_probs = policy(state_tensor)
            new_log_prob = torch.log(new_action_probs.squeeze(0)[action] + 1e-9)
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            policy_loss -= torch.min(surr1, surr2)

            # Value loss
            value_pred = value_network(state_tensor)
            value_loss += (return_ - value_pred) ** 2

        policy_loss /= len(trajectories_states)
        value_loss /= len(trajectories_states)
        (policy_loss - value_loss_coef * value_loss + entropy_coef * policy_entropy).backward()
        policy_optimizer.step()
        value_optimizer.step()

        # Log average episode duration every 1000 episodes
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Average Duration {np.mean(episode_durations[-1000:])}")

    return policy, value_network