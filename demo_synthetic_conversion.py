"""
Demonstration script for converting synthetic data to graph format
and testing with GraphSAGE model
"""

import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

# Import our conversion functions
from convert_synthetic_to_graph import (
    convert_synthetic_dataset_to_graph,
    analyze_converted_data,
    save_converted_data
)

def main():
    print("=== Synthetic Data to Graph Conversion Demo ===\n")
    
    # Step 1: Load synthetic dataset
    print("1. Loading synthetic dataset...")
    try:
        with open('synthetic_dataset.pkl', 'rb') as f:
            synthetic_dataset = pickle.load(f)
        print(f"   ✓ Loaded {len(synthetic_dataset)} synthetic circuits")
    except FileNotFoundError:
        print("   ✗ Error: synthetic_dataset.pkl not found")
        print("   Please run the synthetic-data-generation.ipynb notebook first")
        return
    
    # Show original structure
    sample = synthetic_dataset[0]
    print(f"   Sample structure: {sample.keys()}")
    print(f"   Feature dimensions: {sample['features'].shape}")
    
    class_counts = Counter([s['class'] for s in synthetic_dataset])
    print("   Original class distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = count / len(synthetic_dataset) * 100
        print(f"     Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    # Step 2: Convert to graph format
    print(f"\n2. Converting to graph format...")
    graph_data_list = convert_synthetic_dataset_to_graph(synthetic_dataset)
    print(f"   ✓ Successfully converted {len(graph_data_list)} circuits")
    
    # Step 3: Analyze converted data
    print(f"\n3. Analyzing converted data...")
    analyze_converted_data(graph_data_list)
    
    # Step 4: Save converted data
    print(f"\n4. Saving converted data...")
    save_converted_data(graph_data_list, 'synthetic_circuit_data_classification.pkl')
    
    # Step 5: Test compatibility
    print(f"\n5. Testing PyTorch Geometric compatibility...")
    test_loader = DataLoader(graph_data_list[:16], batch_size=8, shuffle=False)
    
    for batch in test_loader:
        print(f"   ✓ Batch test successful:")
        print(f"     Node features: {batch.x.shape}")
        print(f"     Edge index: {batch.edge_index.shape}")
        print(f"     Targets: {batch.y.shape}")
        print(f"     Graphs in batch: {batch.num_graphs}")
        break
    
    print(f"\n=== Conversion Complete ===")
    print(f"Synthetic data is now ready for GraphSAGE inference!")
    print(f"Files created:")
    print(f"  - synthetic_circuit_data_classification.pkl")
    print(f"  - synthetic_circuit_data_classification_metadata.pkl")

if __name__ == "__main__":
    main()
