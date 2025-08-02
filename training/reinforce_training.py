#!/usr/bin/env python3
"""REINFORCE (Policy Gradient) Training for Refactored Farm Environment"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Add environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))
from custom_env import FarmEnv

class FarmEnvWrapper(gym.Wrapper):
    """Flatten dictionary observation space for compatibility."""
    def __init__(self, env):
        super().__init__(env)
        test_obs, _ = env.reset()
        if isinstance(test_obs, dict):
            self.obs_size = sum(value.flatten().shape[0] if hasattr(value, 'flatten') else 1 
                               for value in test_obs.values())
        else:
            self.obs_size = len(test_obs) if hasattr(test_obs, '__len__') else 1
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=5.0, shape=(self.obs_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            flat_components = []
            for value in obs.values():
                if hasattr(value, 'flatten'):
                    flat_components.append(value.flatten())
                else:
                    flat_components.append(np.array([value]))
            return np.concatenate(flat_components).astype(np.float32)
        return np.array(obs, dtype=np.float32)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    logits = policy(state)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def train_reinforce(episodes=1000, gamma=0.99, lr=0.001, save_path="models/reinforce/reinforce_farm_final.pt"):
    print("🚀 Training REINFORCE...")
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("logs/reinforce", exist_ok=True)

    env = FarmEnvWrapper(FarmEnv())
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            action, log_prob = select_action(policy, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated
            state = next_state
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Policy loss
        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward (last 50): {avg_reward:.1f}")
    torch.save(policy.state_dict(), save_path)
    print("✅ REINFORCE training complete!")
    return policy

def evaluate_reinforce(model_path="models/reinforce/reinforce_farm_final.pt", episodes=10):
    try:
        env = FarmEnvWrapper(FarmEnv())
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        policy = PolicyNetwork(obs_dim, n_actions)
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
        rewards = []
        harvests = []
        for ep in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    action, _ = select_action(policy, state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
            rewards.append(total_reward)
            harvests.append(getattr(env.env, 'harvests_count', 0))
        avg_reward = np.mean(rewards)
        avg_harvests = np.mean(harvests)
        max_reward = np.max(rewards)
        print(f"📊 Results: Avg Reward={avg_reward:.1f}, Max={max_reward:.1f}, Avg Harvests={avg_harvests:.1f}")
        return avg_reward, avg_harvests, max_reward
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 0, 0, 0

def main():
    print("🌾 REINFORCE Farm Training")
    policy = train_reinforce()
    avg_reward, avg_harvests, max_reward = evaluate_reinforce()
    
    # Save results
    from results_tracker import ResultsTracker
    tracker = ResultsTracker()
    tracker.add_result("reinforce", avg_reward, max_reward, avg_harvests)
    
    print("🎬 Ready for visualization - run: python play.py --algo reinforce")
    print("✅ Complete!")

if __name__ == "__main__":
    main() 