import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(description='Connect4 Reinforcement Learning')
    parser.add_argument('--train', action='store_true', help='Train all agents')
    parser.add_argument('--use', action='store_true', help='Use trained agents')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes for training')
    
    args = parser.parse_args()
    
    if args.train:
        from train_agents import main as train_main
        print("Starting training process...")
        train_main()
    elif args.use:
        from use_trained_agents import main as use_main
        print("Using trained agents...")
        use_main()
    else:
        print("No action specified. Use --train to train agents or --use to use trained agents.")
        parser.print_help()

if __name__ == "__main__":
    main()