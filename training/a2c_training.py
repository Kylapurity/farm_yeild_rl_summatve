#!/usr/bin/env python3
"""A2C Training for Refactored Farm Environment"""

import os
import sys
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import gymnasium as gym

# Add environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))
from custom_env import FarmEnv

class FarmEnvWrapper(gym.Wrapper):
    """Flatten dictionary observation space for SB3 compatibility."""
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

def train_a2c():
    """Train A2C on the refactored farm environment."""
    print("🚀 Training A2C...")
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)

    env = FarmEnvWrapper(FarmEnv())
    env = Monitor(env, "logs/a2c/")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.25,
        max_grad_norm=0.5,
        rms_prop_eps=1e-05,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=False,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=0,
        device=device
    )

    eval_env = FarmEnvWrapper(FarmEnv())
    eval_env = Monitor(eval_env, "logs/a2c/eval/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/a2c/",
        eval_freq=25000,
        deterministic=True,
        verbose=0
    )

    print("Training... (this may take a while)")
    model.learn(total_timesteps=100000, callback=eval_callback, progress_bar=False)
    model.save("models/a2c/a2c_farm_final")
    print("✅ A2C training complete!")
    return model

def evaluate_a2c(model_path="models/a2c/a2c_farm_final", episodes=10):
    """Evaluate a trained A2C model on the farm environment."""
    try:
        model = A2C.load(model_path)
        eval_env = FarmEnvWrapper(FarmEnv())
        rewards = []
        harvests = []
        for ep in range(episodes):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if hasattr(action, 'item'):
                    action = action.item()
                elif isinstance(action, np.ndarray):
                    action = int(action[0])
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
            harvests.append(getattr(eval_env.env, 'harvests_count', 0))
        avg_reward = np.mean(rewards)
        avg_harvests = np.mean(harvests)
        max_reward = np.max(rewards)
        print(f"📊 Results: Avg Reward={avg_reward:.1f}, Max={max_reward:.1f}, Avg Harvests={avg_harvests:.1f}")
        return avg_reward, avg_harvests, max_reward
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 0, 0, 0

def main():
    print("🌾 A2C Farm Training")
    model = train_a2c()
    avg_reward, avg_harvests, max_reward = evaluate_a2c()
    
    # Save results
    from results_tracker import ResultsTracker
    tracker = ResultsTracker()
    tracker.add_result("a2c", avg_reward, max_reward, avg_harvests)
    
    print("🎬 Ready for visualization - run: python play.py --algo a2c")
    print("✅ Complete!")

if __name__ == "__main__":
    main()
