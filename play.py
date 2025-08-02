#!/usr/bin/env python3
"""Visualize trained RL agent in the farm environment using OpenGL rendering."""

import argparse
import os
import sys
import numpy as np
import time
from datetime import datetime

# Add environment and rendering paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
from custom_env import FarmEnv
from render import FarmRenderer

# Import RL libraries
from stable_baselines3 import DQN, PPO, A2C
import torch

# For video/GIF recording
try:
    import imageio
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_VIDEO = True
except ImportError:
    print("⚠️  imageio or matplotlib not available. Video recording disabled.")
    HAS_VIDEO = False

# For REINFORCE
class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def flatten_obs(obs):
    if isinstance(obs, dict):
        flat_components = []
        for value in obs.values():
            if hasattr(value, 'flatten'):
                flat_components.append(value.flatten())
            else:
                flat_components.append(np.array([value]))
        return np.concatenate(flat_components).astype(np.float32)
    return np.array(obs, dtype=np.float32)

def load_model(algo, model_path, env):
    if algo == 'dqn':
        return DQN.load(model_path)
    elif algo == 'ppo':
        return PPO.load(model_path)
    elif algo == 'a2c':
        return A2C.load(model_path)
    elif algo == 'reinforce':
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        policy = PolicyNetwork(obs_dim, n_actions)
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
        return policy
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

class VideoRecorder:
    """Records farm environment as video/GIF"""
    def __init__(self, save_path, fps=10):
        self.save_path = save_path
        self.fps = fps
        self.frames = []
        self.crop_colors = ['white', 'lightgreen', 'green', 'red', 'gold']
        self.crop_labels = ['BARE', 'PLANTED', 'GROWING', 'DAMAGED', 'READY']
        
    def capture_frame(self, env, episode, step, reward, action_name):
        """Capture current state as a frame"""
        if not HAS_VIDEO:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Main grid visualization
        grid = env.grid
        moisture = env.moisture
        nutrients = env.nutrients
        
        # Create color map for grid
        cmap = ListedColormap(self.crop_colors)
        im1 = ax1.imshow(grid, cmap=cmap, vmin=0, vmax=4)
        
        # Add text annotations
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                crop_stage = grid[i, j]
                moisture_val = moisture[i, j]
                nutrient_val = nutrients[i, j]
                
                # Color text based on crop stage
                if crop_stage == 0:  # BARE
                    text_color = 'black'
                elif crop_stage == 4:  # READY
                    text_color = 'darkred'
                else:
                    text_color = 'white'
                
                ax1.text(j, i, f'{self.crop_labels[crop_stage][0]}\n{moisture_val:.1f}\n{nutrient_val:.1f}', 
                        ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
        
        ax1.set_title(f'Farm Grid - Episode {episode}, Step {step}')
        ax1.set_xticks(range(grid.shape[1]))
        ax1.set_yticks(range(grid.shape[0]))
        ax1.set_xticklabels(range(grid.shape[1]))
        ax1.set_yticklabels(range(grid.shape[0]))
        
        # Add colorbar
        cbar = plt.colorbar(im1, ax=ax1, ticks=range(5))
        cbar.set_ticklabels(self.crop_labels)
        
        # Statistics panel
        ax2.axis('off')
        stats_text = f"""
        🌾 FARM AGENT PERFORMANCE
        
        Episode: {episode}
        Step: {step}
        Current Reward: {reward:.1f}
        Last Action: {action_name}
        
        📊 CUMULATIVE STATS
        Total Yield: {env.total_yield:.1f}
        Harvests: {env.harvests_count}
        Steps: {env.steps}
        
        🎯 CROP DISTRIBUTION
        Bare: {np.sum(grid == 0)}
        Planted: {np.sum(grid == 1)}
        Growing: {np.sum(grid == 2)}
        Damaged: {np.sum(grid == 3)}
        Ready: {np.sum(grid == 4)}
        """
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        self.frames.append(img)
        plt.close(fig)
    
    def save_video(self):
        """Save frames as GIF"""
        if not HAS_VIDEO or not self.frames:
            return
            
        print(f"🎬 Saving {len(self.frames)} frames as GIF...")
        imageio.mimsave(self.save_path, self.frames, fps=self.fps)
        print(f"✅ Video saved to: {self.save_path}")

def save_performance_data(algo, episode_rewards, episode_harvests, episode_yields, total_steps, model_path=None):
    """Save performance data from visualization session"""
    import json
    from datetime import datetime
    
    if not episode_rewards:
        print("❌ No performance data to save")
        return
    
    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    max_reward = max(episode_rewards)
    avg_harvests = sum(episode_harvests) / len(episode_harvests)
    avg_yield = sum(episode_yields) / len(episode_yields)
    
    # Create model identifier from path
    model_name = "default"
    if model_path:
        # Extract model name from path
        if "/" in model_path:
            model_name = model_path.split("/")[-1]  # Get filename
        else:
            model_name = model_path
    
    # Create performance data
    performance_data = {
        'algorithm': algo,
        'model_path': model_path,
        'model_name': model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'episodes_run': len(episode_rewards),
        'total_steps': total_steps,
        'statistics': {
            'avg_reward': round(avg_reward, 1),
            'max_reward': round(max_reward, 1),
            'avg_harvests': round(avg_harvests, 1),
            'avg_yield': round(avg_yield, 1)
        },
        'episode_details': [
            {
                'episode': i + 1,
                'reward': round(reward, 1),
                'harvests': harvests,
                'yield': round(yield_val, 1)
            }
            for i, (reward, harvests, yield_val) in enumerate(zip(episode_rewards, episode_harvests, episode_yields))
        ]
    }
    
    # Save to file with model name
    os.makedirs('play_results', exist_ok=True)
    safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace('.', '_')
    filename = f"play_results/{algo}_{safe_model_name}_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    # Print summary
    print(f"\n📊 VISUALIZATION PERFORMANCE SUMMARY:")
    print(f"Algorithm: {algo.upper()}")
    print(f"Model: {model_name}")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Total Steps: {total_steps}")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Max Reward: {max_reward:.1f}")
    print(f"Average Harvests: {avg_harvests:.1f}")
    print(f"Average Yield: {avg_yield:.1f}")
    print(f"📁 Performance data saved to: {filename}")
    
    # Update results tracker if available
    try:
        from results_tracker import ResultsTracker
        tracker = ResultsTracker()
        tracker.add_result(algo, avg_reward, max_reward, avg_harvests)
        print("✅ Results also saved to training tracker")
    except ImportError:
        print("ℹ️  Results tracker not available")

def main():
    parser = argparse.ArgumentParser(description="Visualize trained RL agent in the farm environment.")
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'ppo', 'a2c', 'reinforce'],
                        help='Algorithm to visualize (dqn, ppo, a2c, reinforce)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run (default: 5)')
    parser.add_argument('--record', action='store_true', help='Record video/GIF of the session')
    parser.add_argument('--fps', type=int, default=10, help='FPS for video recording (default: 10)')
    args = parser.parse_args()

    # Set default model paths
    default_paths = {
        'dqn': 'models/dqn/dqn_farm_final',
        'ppo': 'models/ppo/ppo_farm_final',
        'a2c': 'models/a2c/a2c_farm_final',
        'reinforce': 'models/reinforce/reinforce_farm_final.pt'
    }
    model_path = args.model_path or default_paths[args.algo]

    # Create environment
    from custom_env import FarmEnv
    env = FarmEnv(render_mode='human')
    obs, _ = env.reset()

    # Load model
    model = load_model(args.algo, model_path, env)

    # Create renderer
    from render import FarmRenderer
    renderer = FarmRenderer(env)
    
    # Performance tracking
    episode_rewards = []
    episode_harvests = []
    episode_yields = []
    total_steps = 0
    
    # Video recording setup
    video_recorder = None
    if args.record:
        if not HAS_VIDEO:
            print("❌ Video recording requires imageio and matplotlib. Install with: pip install imageio matplotlib")
            return
        
        # Create video filename
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace('.', '_')
        video_filename = f"play_results/{algo}_{safe_model_name}_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        os.makedirs('play_results', exist_ok=True)
        video_recorder = VideoRecorder(video_filename, fps=args.fps)
        print(f"🎬 Video recording enabled: {video_filename}")

    # Action names for display
    action_names = ['PLANT', 'IRRIGATE', 'FERTILIZE', 'HARVEST']
    
    def agent_policy(obs):
        flat_obs = flatten_obs(obs)
        if args.algo == 'reinforce':
            with torch.no_grad():
                logits = model(torch.from_numpy(flat_obs).float().unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item()
            return action
        else:
            action, _ = model.predict(flat_obs, deterministic=True)
            if hasattr(action, 'item'):
                return action.item()
            elif isinstance(action, np.ndarray):
                return int(action[0])
            return int(action)

    # Main loop
    episode_count = 0
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    print(f"🎮 Starting visualization for {args.algo.upper()} agent...")
    print(f"📊 Running {args.episodes} episodes")
    print("💡 Press Ctrl+C to stop early")
    
    try:
        while episode_count < args.episodes:
            action = agent_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Record frame if video recording is enabled
            if video_recorder:
                action_name = action_names[action] if 0 <= action < len(action_names) else f"UNKNOWN({action})"
                video_recorder.capture_frame(env, episode_count + 1, episode_steps, episode_reward, action_name)
            
            renderer.render_scene()
            
            if terminated or truncated:
                # Episode finished - save performance data
                episode_rewards.append(episode_reward)
                episode_harvests.append(env.harvests_count)
                episode_yields.append(env.total_yield)
                
                print(f"📈 Episode {episode_count + 1}: Reward={episode_reward:.1f}, "
                      f"Harvests={env.harvests_count}, Yield={env.total_yield:.1f}")
                
                episode_count += 1
                obs, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
    
    # Save video if recording was enabled
    if video_recorder:
        video_recorder.save_video()
    
    # Save performance data
    save_performance_data(args.algo, episode_rewards, episode_harvests, episode_yields, total_steps, model_path)

if __name__ == '__main__':
    main()