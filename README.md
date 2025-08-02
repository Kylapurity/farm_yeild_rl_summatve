# Smart Farm Management with Reinforcement Learning

This project implements an intelligent farming system using reinforcement learning to optimize agricultural decision-making. The system models the complex dynamics of crop management, including planting, irrigation, fertilization, pest control, and harvesting to maximize yield while efficiently managing resources.

Reinforcement learning agents—including **Deep Q-Networks (DQN)**, **Proximal Policy Optimization (PPO)**, **Actor-Critic (A2C)**, and **REINFORCE**—are trained and compared in this custom farming environment built with OpenGL visualization.

## Problem Statement

Modern agriculture faces increasing complexity in decision-making due to climate variability, resource constraints, and the need for sustainable practices. Farmers must make optimal timing decisions for planting, irrigating, fertilizing, pest management, and harvesting to maximize yield while minimizing resource waste and environmental impact.

This simulation framework enables experimentation with intelligent agents that can learn optimal farming strategies under various environmental conditions and resource constraints.

## Project Structure

```
farm_management_rl/
├── environment/
│   ├── farming_env.py          # Custom Gymnasium environment implementation
│   ├── rendering.py            # OpenGL visualization components
│   └── environment_gif.gif     # GIF of farming actions in the environment
├── training/
│   ├── dqn_training.py         # Training script for DQN using Stable-Baselines3
│   ├── ppo_training.py         # Training script for PPO using Stable-Baselines3
│   ├── a2c_training.py         # Training script for A2C using Stable-Baselines3
│   ├── reinforce_training.py   # Custom REINFORCE implementation
│   ├── training_progress.png   # Graph showing training performance
│   └── algorithm_comparison.png # Graph comparing all algorithm performances
├── models/
│   ├── dqn/                    # Saved DQN models
│   ├── ppo/                    # Saved PPO models
│   ├── a2c/                    # Saved A2C models
│   └── reinforce/              # Saved REINFORCE models
├── evaluation/
│   ├── evaluate_agents.py      # Evaluation scripts for trained models
│   ├── performance_metrics.py  # Performance analysis tools
│   └── results/                # Evaluation results and graphs
├── main.py                     # Entry point for running experiments
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Environment Description

### Agent
The agent represents a **Smart Farmer** who navigates through a 5×5 farm grid, making decisions about crop management at each location. The farmer must optimize the entire farm by managing multiple crop growth stages simultaneously.

### Action Space
The environment provides four discrete actions:

| Action | ID | Description | Requirements |
|--------|----| ------------|--------------|
| **PLANT** | 0 | Converts bare soil to planted crops | Cell must be BARE (0) |
| **IRRIGATE** | 1 | Increases moisture levels for crops | Cell must have planted/growing crops |
| **FERTILIZE** | 2 | Increases nutrient levels for crops | Cell must have planted/growing crops |
| **HARVEST** | 3 | Collects ready crops and converts back to bare soil | Cell must be READY (4) |

### State Space
The state representation includes multiple layers of information:

**Crop States (5×5 grid):**
- `0 = BARE` - Empty soil ready for planting
- `1 = PLANTED` - Seeds planted, beginning growth
- `2 = GROWING` - Crops actively growing
- `3 = DAMAGED` - Crops damaged by pests
- `4 = READY` - Crops ready for harvest

**Environmental Factors:**
- **Moisture (5×5):** Water levels (0.0 to 1.0) affecting crop growth
- **Nutrients (5×5):** Soil nutrient levels (0.0 to 1.0) affecting crop growth  
- **Pests (5×5):** Binary pest presence (0 or 1) that can damage crops
- **Cell Position (2):** Coordinates [x, y] of the last cell acted upon

### Reward Structure
The reward system encourages efficient farming practices:

**Step-based Rewards:**
- Each step: **-3** (encourages efficiency and timely decisions)
- Progress toward goals (≤2 tiles): **+3** (planting, fertilizing, watering)
- Very close to goals (≤1 tile): **+5** (harvesting actions)
- Reaching optimal state: **+8**

**Final Episode Reward:**
```
R_final = 0.2 × total_yield + 6.0 × harvest_count
```

This formula balances immediate harvest rewards with overall farm productivity.

### Environment Visualization

The environment features a sophisticated 3D OpenGL visualization system:

- **5×5 Farm Grid:** Clearly defined farming plots
- **Color-coded Crop Stages:** 
  - Brown: Bare soil
  - Light Green: Planted seeds
  - Dark Green: Growing crops
  - Yellow: Ready for harvest
  - Red: Damaged crops
- **Environmental Indicators:**
  - Blue droplets: Moisture levels
  - Green spheres: Nutrient availability
  - Red cones: Pest presence
- **Real-time Updates:** Dynamic visualization during training and evaluation

## Reinforcement Learning Algorithms

### Deep Q-Network (DQN)
**Architecture:** Multi-Layer Perceptron with [128, 128] hidden layers
**Key Hyperparameters:**
- Learning Rate: 0.0002
- Buffer Size: 20,000
- Exploration Fraction: 0.05
- Batch Size: 64
- Gamma: 0.99
- Training Steps: 200,000

**Performance:** Achieved the highest average reward of **6773.7**, demonstrating superior learning capability for discrete action selection in farming scenarios.

### Proximal Policy Optimization (PPO)
**Architecture:** Multi-Layer Perceptron with [128, 128] hidden layers
**Key Hyperparameters:**
- Learning Rate: 0.0003
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip Range: 0.2
- Entropy Coefficient: 0.01

**Performance:** Second-best performance with average reward of **558**, showing good stability and policy improvement.

### Actor-Critic (A2C)
**Architecture:** Multi-Layer Perceptron with [128, 128] hidden layers
**Key Hyperparameters:**
- Learning Rate: 0.0007
- Gamma: 0.99
- Entropy Coefficient: 0.01
- Value Function Coefficient: 0.5
- Max Gradient Norm: 0.5

**Performance:** Achieved **617** average reward with fastest convergence, balancing learning speed with value estimation accuracy.

### REINFORCE (Custom Implementation)
**Architecture:** Multi-Layer Perceptron with [128, 128] hidden layers
**Key Hyperparameters:**
- Learning Rate: 0.0002
- Gamma: 0.99
- Episodes: 1000
- Policy: Categorical distribution

**Performance:** Baseline performance with **96** average reward, showing improvement from initial negative rewards but limited by high variance.

## Evaluation Metrics

The project evaluates algorithms across multiple dimensions:

### Performance Metrics
- **Cumulative Reward:** Total reward achieved per episode
- **Harvest Efficiency:** Number of successful harvests per episode
- **Resource Utilization:** Optimal use of water and nutrients
- **Training Stability:** Consistency of learning across episodes
- **Convergence Speed:** Episodes required to reach stable performance
- **Generalization:** Performance on unseen farm configurations

### Effectiveness Score
```
Effectiveness = (Total Yield × 0.3) + (Harvest Count × 0.4) + (Resource Efficiency × 0.3)
```

## Installation and Setup

### Requirements
```bash
# Core dependencies
pip install gymnasium
pip install stable-baselines3[extra]
pip install torch torchvision
pip install numpy matplotlib
pip install pygame PyOpenGL PyOpenGL_accelerate
pip install pillow imageio

# Optional for advanced visualization
pip install tensorboard
```

### Quick Start
```bash
# Clone the repository
git clone [YOUR_REPOSITORY_LINK]
cd smart_farming_rl

# Install dependencies
pip install -r requirements.txt

# Train all algorithms
python main.py --train-all

# Evaluate specific algorithm
python main.py --evaluate --algorithm dqn

# Run environment visualization
python environment/farming_env.py --demo
```

## Running Experiments

### Training Individual Algorithms
```bash
# Train DQN agent
python training/dqn_training.py

# Train PPO agent  
python training/ppo_training.py

# Train A2C agent
python training/a2c_training.py

# Train REINFORCE agent
python training/reinforce_training.py
```

### Hyperparameter Optimization
The project includes comprehensive hyperparameter tuning results showing the impact of various parameters on performance:

- **Learning Rate:** Optimal values varied by algorithm (0.0002-0.0007)
- **Gamma:** High discount factor (0.99) crucial for long-term farming rewards
- **Buffer Size:** 20,000 provided optimal balance of diversity and memory efficiency
- **Exploration:** Epsilon-greedy with final value 0.05 worked best for DQN

### Performance Comparison
Run the complete evaluation suite:
```bash
python evaluation/evaluate_agents.py --all-algorithms --episodes 100
```

## Results and Insights

### Algorithm Performance Ranking
1. **DQN (447.7)** -Fast convergence with balanced actor-critic learning
2. **A2C (617)** - Superior discrete action selection and experience replay
3. **PPO (558)** - Stable training with good policy optimization
4. **REINFORCE (96)** - Baseline performance with high variance issues

### Key Findings
- **DQN excelled** due to discrete action space compatibility and effective experience replay
- **A2C provided fastest convergence** through balanced actor-critic architecture
- **PPO demonstrated stability** but with lower peak performance
- **REINFORCE required significant variance reduction** for competitive performance

### Generalization Testing
Testing across different farm configurations revealed:
- DQN: Excellent generalization across varied initial conditions
- A2C: Good adaptation with slight performance variation
- PPO: Moderate generalization with some sensitivity to starting states
- REINFORCE: Poor generalization due to training instability

## Future Enhancements

### Algorithmic Improvements
- Implement **Prioritized Experience Replay** for DQN
- Add **Dueling DQN architecture** for better value estimation
- Explore **Multi-Agent scenarios** for cooperative farming
- Integrate **Hierarchical RL** for long-term planning

### Environment Extensions
- **Weather Systems:** Dynamic weather affecting crop growth
- **Market Economics:** Price fluctuations affecting harvest values  
- **Crop Varieties:** Different crops with unique growth patterns
- **Seasonal Cycles:** Long-term agricultural planning
- **Sustainability Metrics:** Environmental impact considerations

### Technical Enhancements
- **Curriculum Learning:** Progressive difficulty increase
- **Transfer Learning:** Apply policies across different farm types
- **Real-time Adaptation:** Dynamic parameter adjustment
- **3D Visualization:** Enhanced OpenGL rendering with realistic farm models

## Links and Resources

### Project Links
- **Video Demonstration:** https://veed.io/view/55f6d580-17bf-47c4-a256-02cd074473fd
- **Agent Live Demo:**  https://veed.io/view/1018e830-d975-4853-8fbd-ec618589646d
## Contributing

We welcome contributions to improve the farming simulation and RL implementations:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request




---

**Project Status:** Active Development | **Last Updated:** August 2024 | **Version:** 1.0.0
