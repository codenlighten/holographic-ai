"""
Plot training progress from checkpoint
"""
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_training_info(info_file):
    """Create plots from training info"""
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    print("Training Information:")
    print("="*60)
    for key, value in info.items():
        if key != 'config':
            print(f"  {key}: {value}")
    
    print("\nConfiguration:")
    for key, value in info['config'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("info_file", type=str, help="Path to training_info.json")
    args = parser.parse_args()
    
    plot_training_info(args.info_file)
