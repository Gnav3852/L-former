#!/usr/bin/env python3
"""
Test script to verify dataset balance before training
"""

from data.real_text_dataset import RealTextDataset
from config import ModelConfig

def test_dataset_balance():
    """Test the dataset balance"""
    print("Testing dataset balance...")
    
    # Create config
    config = ModelConfig.tiny()
    config.n_tools = 3
    
    # Create dataset
    dataset = RealTextDataset(
        n_tools=config.n_tools,
        n_samples=1000,
        max_seq_len=100
    )
    
    # Print statistics
    dataset.print_stats()
    
    # Test a few examples
    print("\nTesting first few examples:")
    for i in range(5):
        example = dataset[i]
        print(f"Example {i}: Type={example['type']}, Tool={example['tool_labels'] if example['tool_labels'] is not None else 'N/A'}")

if __name__ == "__main__":
    test_dataset_balance()
