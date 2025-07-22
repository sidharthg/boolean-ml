# Synthetic Data to GraphSAGE Conversion

This guide explains how to convert synthetic Boolean circuit data into the graph data structure expected by the GraphSAGE baseline classification model.

## Overview

The conversion process transforms PyEDA Boolean expressions into PyTorch Geometric `Data` objects with the same structure as the preprocessing pipeline used for real circuit data.

## Files Created

1. **`convert_synthetic_to_graph.py`** - Main conversion script
2. **`demo_synthetic_conversion.py`** - Demonstration script
3. **Updated `synthetic-data-generation.ipynb`** - Enhanced feature extraction

## Step-by-Step Process

### 1. Generate Synthetic Data
First, run the synthetic data generation notebook:
```bash
# Run the synthetic-data-generation.ipynb notebook
# This creates synthetic_dataset.pkl
```

### 2. Convert to Graph Format
Run the conversion script:
```python
python convert_synthetic_to_graph.py
```

Or use the demo script:
```python
python demo_synthetic_conversion.py
```

### 3. Use with GraphSAGE Model
The converted data can now be used with the GraphSAGE model:
```python
import pickle
from torch_geometric.data import DataLoader

# Load converted data
with open('synthetic_circuit_data_classification.pkl', 'rb') as f:
    synthetic_graph_data = pickle.load(f)

# Create DataLoader
loader = DataLoader(synthetic_graph_data, batch_size=16, shuffle=False)

# Use with trained GraphSAGE model for inference
# (Load model from Graphsage-baseline-classification-v2.ipynb)
```

## Data Structure Details

### Input Format (Synthetic Data)
```python
{
    'circuit': PyEDA Boolean expression,
    'bias': float (0.0 to 1.0),
    'class': int (0, 1, or 2),
    'features': numpy array (13-dimensional)
}
```

### Output Format (Graph Data)
```python
Data(
    x=torch.tensor,           # Node features [num_nodes, 13]
    edge_index=torch.tensor,  # Graph connectivity [2, num_edges]
    y=torch.tensor,           # Class label (0, 1, or 2)
    num_nodes=int,            # Number of nodes
    hamming_weight=int,       # Approximate hamming weight
    input_count=int,          # Number of primary inputs
    slice_label=str,          # Identifier
    circuit_name=str          # Circuit name
)
```

## Feature Extraction (13 dimensions)

The conversion extracts features matching the GraphSAGE preprocessing:

1. **Gate Type One-Hot (9 dims)**: INPUT, OUTPUT, AND, NAND, OR, NOR, NOT, XOR, XNOR
2. **In-degree (1 dim)**: Number of incoming edges
3. **Out-degree (1 dim)**: Number of outgoing edges  
4. **Influence Value (1 dim)**: Computed connectivity metric
5. **Primary Input Flag (1 dim)**: 1 if primary input, 0 otherwise

## Graph Construction Process

1. **Expression Tree Extraction**: Parse PyEDA Boolean expressions into directed graphs
2. **Node Type Assignment**: Map operators to standard gate types
3. **Context Extension**: Include connected nodes for richer representation
4. **Edge Creation**: Convert to undirected edges for GraphSAGE
5. **Feature Computation**: Extract 13-dimensional node features

## Class Mapping

Classes are preserved from synthetic data generation:
- **Class 0**: Ultra-sparse circuits (bias < 0.01)
- **Class 1**: Sparse circuits (0.01 ≤ bias < 0.06)  
- **Class 2**: Dense circuits (bias ≥ 0.06)

## Usage with GraphSAGE

After conversion, the data can be directly used with the trained GraphSAGE model:

```python
# Load trained model (from Graphsage-baseline-classification-v2.ipynb)
model.load_state_dict(torch.load('best_imbalanced_model.pth'))
model.eval()

# Get predictions
with torch.no_grad():
    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.batch)
        predictions = logits.argmax(dim=1)
        probabilities = F.softmax(logits, dim=1)
```

## Files Generated

- `synthetic_circuit_data_classification.pkl` - Converted graph data
- `synthetic_circuit_data_classification_metadata.pkl` - Dataset metadata

## Verification

The conversion script includes analysis functions to verify:
- Feature dimensions match (13D)
- Data types are correct (torch.float, torch.long)
- Class distribution is preserved
- Graph structure is valid for PyTorch Geometric

## Troubleshooting

**Error: "synthetic_dataset.pkl not found"**
- Run the synthetic-data-generation.ipynb notebook first

**Feature dimension mismatch**
- Check that extract_circuit_features() returns exactly 13 dimensions

**Graph construction failures**
- Complex expressions may fail - the script handles these gracefully

**Memory issues with large datasets**
- Process data in smaller batches if needed
