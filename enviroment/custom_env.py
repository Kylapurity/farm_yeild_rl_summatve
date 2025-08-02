import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import random

class CropStage(Enum):
    """Crop growth stages with explicit integer values"""
    BARE = 0
    PLANTED = 1
    GROWING = 2
    DAMAGED = 3
    READY = 4

class Action(Enum):
    """Main farming actions only"""
    PLANT = 0
    IRRIGATE = 1
    FERTILIZE = 2
    HARVEST = 3

class FarmEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 5
        self.render_mode = render_mode
        
        # Observation space: grid of crop stages + soil conditions
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=4, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            'moisture': spaces.Box(low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32),
            'nutrients': spaces.Box(low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32),
            'pests': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            'cell': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        })
        
        self.action_space = spaces.Discrete(len(Action))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.full((self.grid_size, self.grid_size), CropStage.BARE.value, dtype=np.int32)
        self.moisture = np.random.uniform(0.7, 0.9, (self.grid_size, self.grid_size))
        self.nutrients = np.random.uniform(0.7, 0.9, (self.grid_size, self.grid_size))
        self.pests = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.total_yield = 0
        self.steps = 0
        self.last_action = None
        self.harvests_count = 0
        self.last_cell = [0, 0]
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Reset action tracking
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            'grid': self.grid,
            'moisture': self.moisture,
            'nutrients': self.nutrients,
            'pests': self.pests,
            'cell': np.array(self.last_cell, dtype=np.int32)
        }

    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        self.last_action = action
        
        # Select a random cell to act on
        x, y = np.random.randint(0, self.grid_size, size=2)
        self.last_cell = [x, y]
        
        # Track action for results
        if not hasattr(self, 'action_counts'):
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # PLANT, IRRIGATE, FERTILIZE, HARVEST
        self.action_counts[action] += 1
        
        if action == Action.PLANT.value:
            if self.grid[x, y] == CropStage.BARE.value:
                self.grid[x, y] = CropStage.PLANTED.value
                reward += 5.0  # Balanced reward for planting
                info['planted'] = True
            else:
                reward -= 0.2  # Small penalty
                
        elif action == Action.IRRIGATE.value:
            if self.grid[x, y] in [CropStage.PLANTED.value, CropStage.GROWING.value]:
                old_moisture = self.moisture[x, y]
                self.moisture[x, y] = min(1.0, self.moisture[x, y] + 0.4)  # More moisture
                if self.moisture[x, y] > old_moisture:
                    reward += 4.0  # Balanced reward for proper irrigation
                    info['irrigated'] = True
            else:
                self.moisture[x, y] = min(1.0, self.moisture[x, y] + 0.4)
                reward -= 0.2  # Small penalty
            
        elif action == Action.FERTILIZE.value:
            if self.grid[x, y] in [CropStage.PLANTED.value, CropStage.GROWING.value]:
                old_nutrients = self.nutrients[x, y]
                self.nutrients[x, y] = min(1.0, self.nutrients[x, y] + 0.4)  # More nutrients
                if self.nutrients[x, y] > old_nutrients:
                    reward += 4.0  # Balanced reward for proper fertilizing
                    info['fertilized'] = True
            else:
                self.nutrients[x, y] = min(1.0, self.nutrients[x, y] + 0.4)
                reward -= 0.2  # Small penalty
            
        elif action == Action.HARVEST.value:
            if self.grid[x, y] == CropStage.READY.value:
                quality_factor = (self.moisture[x, y] + self.nutrients[x, y]) / 2
                base_yield = 25.0  # Balanced base yield
                yield_amount = base_yield * (0.8 + quality_factor * 0.4)
                reward += yield_amount * 1.0  # Balanced reward for correct harvest
                self.total_yield += yield_amount
                self.harvests_count += 1
                if self.harvests_count == 1:
                    reward += 25.0  # Balanced first harvest bonus
                harvest_bonus = self.harvests_count * 5.0  # Balanced harvest bonus
                reward += harvest_bonus
                self.grid[x, y] = CropStage.BARE.value
                self.moisture[x, y] = 0.8
                self.nutrients[x, y] = 0.7
                info['harvested'] = True
                info['yield'] = yield_amount
                info['harvest_bonus'] = harvest_bonus
            elif self.grid[x, y] in [CropStage.PLANTED.value, CropStage.GROWING.value]:
                reward -= 0.3  # Small penalty for early harvest
            else:
                reward -= 0.1  # Very small penalty for harvesting nothing
        
        # Natural processes
        progressed = self._update_environment()
        reward += progressed * 0.6  # Balanced reward for progressing crops
        
        # Maintenance and efficiency bonuses
        reward += self._calculate_maintenance_bonus()
        reward += self._calculate_efficiency_bonus()
        reward += self._calculate_action_penalty()
        
        self.steps += 1
        if self.steps >= 50:  # Reduced to 50 steps
            terminated = True
            info['total_yield'] = self.total_yield
            info['harvests_count'] = self.harvests_count
            info['action_counts'] = getattr(self, 'action_counts', {0: 0, 1: 0, 2: 0, 3: 0})
            final_bonus = self.total_yield * 0.2 + self.harvests_count * 6.0  # Balanced final bonus for shorter episodes
            reward += final_bonus
            info['final_bonus'] = final_bonus
        
        return self._get_obs(), reward, terminated, False, info

    def _calculate_maintenance_bonus(self):
        bonus = 0
        ready_count = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == CropStage.PLANTED.value:
                    bonus += 0.2  # Balanced bonus for planted crops
                elif self.grid[x, y] == CropStage.GROWING.value:
                    bonus += 0.3  # Balanced bonus for growing crops
                elif self.grid[x, y] == CropStage.READY.value:
                    ready_count += 1
        if ready_count > 3:
            bonus -= ready_count * 0.4  # Balanced penalty for not harvesting
        elif ready_count > 0:
            bonus += ready_count * 1.5  # Balanced bonus for having ready crops
        return bonus

    def _calculate_efficiency_bonus(self):
        bonus = 0
        planted = np.sum(self.grid == CropStage.PLANTED.value)
        growing = np.sum(self.grid == CropStage.GROWING.value)
        ready = np.sum(self.grid == CropStage.READY.value)
        total_crops = planted + growing + ready
        if total_crops >= 8:
            bonus += 0.8  # Balanced bonus for many crops
        elif total_crops >= 5:
            bonus += 0.5
        elif total_crops >= 3:
            bonus += 0.3
        stages_used = 0
        if planted > 0: stages_used += 1
        if growing > 0: stages_used += 1
        if ready > 0: stages_used += 1
        bonus += stages_used * 0.6  # Balanced bonus for diversity
        if ready >= 6:
            bonus -= 0.8  # Balanced penalty for too many ready crops
        elif ready >= 3:
            bonus -= 0.2
        return bonus

    def _calculate_action_penalty(self):
        penalty = 0
        x, y = self.last_cell
        if (self.last_action == Action.PLANT.value and self.grid[x, y] != CropStage.BARE.value):
            penalty -= 0.5  # Reduced penalty
        return penalty

    def _update_environment(self):
        progressed = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (self.grid[x, y] in [CropStage.PLANTED.value, CropStage.GROWING.value] and self.grid[x, y] != CropStage.DAMAGED.value):
                    growth_prob = min(self.moisture[x, y], self.nutrients[x, y]) * 1.2  # Increased growth probability for shorter episodes
                    if random.random() < growth_prob:
                        if self.grid[x, y] == CropStage.PLANTED.value:
                            self.grid[x, y] = CropStage.GROWING.value
                            progressed += 1
                        elif self.grid[x, y] == CropStage.GROWING.value:
                            self.grid[x, y] = CropStage.READY.value
                            progressed += 1
                if random.random() < 0.005:  # Reduced pest probability
                    self.pests[x, y] = 1
                if self.pests[x, y] and self.grid[x, y] > CropStage.BARE.value:
                    if random.random() < 0.05:  # Reduced pest damage
                        self.grid[x, y] = CropStage.DAMAGED.value
                self.moisture[x, y] = max(0.3, self.moisture[x, y] - 0.002)  # Slower depletion
                self.nutrients[x, y] = max(0.3, self.nutrients[x, y] - 0.001)  # Slower depletion
        return progressed

    def render(self):
        if self.render_mode == "human":
            print(f"\nStep: {self.steps}, Total Yield: {self.total_yield:.2f}")
            print(f"Harvests: {self.harvests_count}")
            print("Grid (0=Bare, 1=Planted, 2=Growing, 3=Damaged, 4=Ready):")
            for i in range(self.grid_size):
                row = ""
                for j in range(self.grid_size):
                    row += f" {self.grid[i, j]}  "
                print(row)
            print("-" * 20)
    
    def get_last_action(self):
        return self.last_action