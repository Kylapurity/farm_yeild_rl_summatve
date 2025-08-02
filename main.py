#!/usr/bin/env python3
"""Main experiment runner for all RL algorithms and visualization."""

import argparse
import subprocess
import sys
import os

def run_training(algo):
    """Run training for a specific algorithm"""
    if algo == 'dqn':
        subprocess.run([sys.executable, 'dqn_training.py'])
    elif algo == 'ppo':
        subprocess.run([sys.executable, 'ppo_training.py'])
    elif algo == 'a2c':
        subprocess.run([sys.executable, 'a2c_training.py'])
    elif algo == 'reinforce':
        subprocess.run([sys.executable, 'reinforce_training.py'])
    else:
        print(f"Unknown algorithm: {algo}")

def run_visualization(algo):
    """Run visualization for a specific algorithm"""
    subprocess.run([sys.executable, 'play.py', '--algo', algo])

def show_results():
    """Show training results"""
    from results_tracker import ResultsTracker
    tracker = ResultsTracker()
    tracker.display_results()

def main():
    parser = argparse.ArgumentParser(description="Farm RL Experiment Runner")
    parser.add_argument('--train', type=str, choices=['dqn', 'ppo', 'a2c', 'reinforce', 'all'], 
                        help='Train algorithm')
    parser.add_argument('--visualize', type=str, choices=['dqn', 'ppo', 'a2c', 'reinforce'], 
                        help='Visualize algorithm')
    parser.add_argument('--results', action='store_true', help='Show training results')
    parser.add_argument('--test', action='store_true', help='Test random agent')
    args = parser.parse_args()

    if args.train:
        if args.train == 'all':
            print("🚀 Training all algorithms...")
            for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
                print(f"\n📊 Training {algo.upper()}...")
                run_training(algo)
            print("\n✅ All training complete!")
        else:
            run_training(args.train)
    
    if args.visualize:
        run_visualization(args.visualize)
    
    if args.results:
        show_results()
    
    if args.test:
        subprocess.run([sys.executable, 'test_random_agent.py'])
    
    # If no arguments provided, show help
    if not any([args.train, args.visualize, args.results, args.test]):
        parser.print_help()
        print("\n💡 Quick Examples:")
        print("  python main.py --train all          # Train all algorithms")
        print("  python main.py --train dqn          # Train DQN only")
        print("  python main.py --visualize dqn      # Visualize DQN")
        print("  python main.py --results            # Show results")
        print("  python main.py --test               # Test random agent")

if __name__ == '__main__':
    main() 