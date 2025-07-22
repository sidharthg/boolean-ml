import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from pyeda.inter import *
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import random

def extract_expression_tree(expr, graph=None, node_counter=None, parent_id=None):
    """
    Recursively extract the expression tree from a PyEDA Boolean expression
    and build a NetworkX graph representation
    """
    if graph is None:
        graph = nx.DiGraph()
        node_counter = {'count': 0}
    
    # Create unique node ID
    current_id = f"node_{node_counter['count']}"
    node_counter['count'] += 1
    
    if isinstance(expr, Variable):
        # This is a variable (primary input)
        var_name = str(expr)
        graph.add_node(current_id, 
                      gate_type='INPUT', 
                      is_primary_input=True,
                      is_primary_output=False,
                      variable_name=var_name)
        return current_id, graph
    
    elif hasattr(expr, 'op'):
        # This is an operator node
        op_type = expr.op
        
        # Map PyEDA operators to standard gate types
        gate_type_mapping = {
            'and': 'AND',
            'or': 'OR',
            'not': 'NOT',
            'xor': 'XOR',
            'nor': 'NOR',
            'nand': 'NAND',
            'xnor': 'XNOR'
        }
        
        gate_type = gate_type_mapping.get(op_type, op_type.upper())
        
        # Add the operator node
        graph.add_node(current_id,
                      gate_type=gate_type,
                      is_primary_input=False,
                      is_primary_output=False)
        
        # Process children
        if hasattr(expr, 'xs'):
            # Multiple operands (AND, OR, XOR, etc.)
            for child_expr in expr.xs:
                child_id, graph = extract_expression_tree(child_expr, graph, node_counter)
                graph.add_edge(child_id, current_id)
        elif hasattr(expr, 'x'):
            # Single operand (NOT)
            child_id, graph = extract_expression_tree(expr.x, graph, node_counter)
            graph.add_edge(child_id, current_id)
        
        return current_id, graph
    
    else:
        # Handle other expression types
        graph.add_node(current_id,
                      gate_type='UNKNOWN',
                      is_primary_input=False,
                      is_primary_output=False)
        return current_id, graph

def add_output_node(graph, output_node_id):
    """Add an output node and mark the final computation node as feeding into it"""
    output_id = "output_node"
    graph.add_node(output_id,
                  gate_type='OUTPUT',
                  is_primary_input=False,
                  is_primary_output=True)
    graph.add_edge(output_node_id, output_id)
    return graph

def extract_primary_inputs(graph):
    """Extract primary input nodes from the graph"""
    primary_inputs = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('is_primary_input', False):
            primary_inputs.append(node_id)
    return primary_inputs

def extend_graph_context(graph, primary_inputs):
    """
    Extend the graph context similar to the preprocessing pipeline
    Include primary inputs + connected gates for richer context
    """
    extended_nodes = set(primary_inputs)
    
    # Add immediate successors and their successors for richer context
    for pi_node in primary_inputs:
        successors = list(graph.successors(pi_node))
        extended_nodes.update(successors)
        
        # Add second-hop neighbors (limited to avoid explosion)
        for succ in successors:
            second_hop = list(graph.successors(succ))
            extended_nodes.update(second_hop[:3])  # Limit to 3 per successor
    
    return list(extended_nodes)

def extract_circuit_features(graph, valid_nodes, primary_inputs):
    """
    Extract 13-dimensional features matching the GraphSAGE preprocessing pipeline
    """
    gate_type_mapping = {
        'INPUT': 0, 'OUTPUT': 1, 'AND': 2, 'NAND': 3, 
        'OR': 4, 'NOR': 5, 'NOT': 6, 'XOR': 7, 'XNOR': 8
    }
    
    node_features = []
    
    for node_id in valid_nodes:
        feature_vector = []
        node_attrs = graph.nodes[node_id]
        
        # 1-9. Gate type one-hot encoding (9 dimensions)
        gate_type = node_attrs.get('gate_type', 'UNKNOWN')
        gate_type_onehot = [0] * len(gate_type_mapping)
        if gate_type in gate_type_mapping:
            gate_type_onehot[gate_type_mapping[gate_type]] = 1
        feature_vector.extend(gate_type_onehot)
        
        # 10. In-degree
        in_degree = graph.in_degree(node_id)
        feature_vector.append(in_degree)
        
        # 11. Out-degree
        out_degree = graph.out_degree(node_id)
        feature_vector.append(out_degree)
        
        # 12. Influence value (for synthetic data, use a computed metric)
        # For primary inputs, use a measure based on their connectivity
        if node_id in primary_inputs:
            # Use fan-out as a proxy for influence
            influence_value = float(out_degree) / max(1, len(primary_inputs))
        else:
            # For intermediate nodes, use a combination of in/out degree
            influence_value = float(in_degree * out_degree) / max(1, len(valid_nodes))
        feature_vector.append(influence_value)
        
        # 13. Is primary input indicator
        is_primary_input = 1 if node_id in primary_inputs else 0
        feature_vector.append(is_primary_input)
        
        node_features.append(feature_vector)
    
    return np.array(node_features, dtype=np.float32)

def create_pytorch_geometric_data(circuit_expr, bias, class_label, circuit_id):
    """
    Convert a PyEDA Boolean expression to PyTorch Geometric Data format
    """
    # Extract expression tree and build graph
    output_node_id, graph = extract_expression_tree(circuit_expr)
    
    # Add output node to match preprocessing pipeline
    graph = add_output_node(graph, output_node_id)
    
    # Extract primary inputs
    primary_inputs = extract_primary_inputs(graph)
    
    if not primary_inputs:
        print(f"Warning: No primary inputs found for circuit {circuit_id}")
        return None
    
    # Extend context similar to preprocessing
    valid_nodes = extend_graph_context(graph, primary_inputs)
    
    # Create subgraph with extended node set
    slice_subgraph = graph.subgraph(valid_nodes).copy()
    
    # Create node mapping for consistent indexing
    node_to_idx = {node: idx for idx, node in enumerate(valid_nodes)}
    
    # Extract node features
    node_features = extract_circuit_features(graph, valid_nodes, primary_inputs)
    
    # Create edge index (undirected for GraphSAGE)
    edges = []
    for edge in slice_subgraph.edges():
        if edge[0] in node_to_idx and edge[1] in node_to_idx:
            src_idx = node_to_idx[edge[0]]
            dst_idx = node_to_idx[edge[1]]
            edges.append([src_idx, dst_idx])
            edges.append([dst_idx, src_idx])  # Make undirected
    
    if edges:
        edge_index = np.array(edges).T
    else:
        edge_index = np.array([[], []], dtype=np.int64)
    
    # Calculate input count and hamming weight approximation
    input_count = len(primary_inputs)
    # Use bias to approximate hamming weight for consistency
    total_space = 2 ** input_count
    approximate_hamming_weight = int(bias * total_space)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(class_label, dtype=torch.long),
        num_nodes=len(valid_nodes),
        hamming_weight=approximate_hamming_weight,
        input_count=input_count,
        slice_label=f"synthetic_{circuit_id}",
        circuit_name=f"synthetic_circuit_{circuit_id}"
    )
    
    return data

def convert_synthetic_dataset_to_graph(synthetic_dataset):
    """
    Convert the entire synthetic dataset to PyTorch Geometric format
    """
    graph_data_list = []
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"Converting {len(synthetic_dataset)} synthetic circuits to graph format...")
    
    for i, sample in enumerate(synthetic_dataset):
        try:
            circuit_expr = sample['circuit']
            bias = sample['bias']
            class_label = sample['class']
            
            # Convert to graph format
            graph_data = create_pytorch_geometric_data(circuit_expr, bias, class_label, i)
            
            if graph_data is not None:
                graph_data_list.append(graph_data)
                successful_conversions += 1
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(synthetic_dataset)} circuits...")
            else:
                failed_conversions += 1
                
        except Exception as e:
            print(f"Error converting circuit {i}: {e}")
            failed_conversions += 1
    
    print(f"\nConversion complete:")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    
    return graph_data_list

def analyze_converted_data(graph_data_list):
    """
    Analyze the converted graph data to ensure it matches expected format
    """
    if not graph_data_list:
        print("No data to analyze")
        return
    
    print(f"\n{'='*50}")
    print("CONVERTED DATA ANALYSIS")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"Total samples: {len(graph_data_list)}")
    
    # Class distribution
    class_counts = Counter([data.y.item() for data in graph_data_list])
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = count / len(graph_data_list) * 100
        print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    # Feature dimensions
    sample_data = graph_data_list[0]
    print(f"\nFeature dimensions: {sample_data.x.shape[1]}")
    print(f"Sample node count range: {min([d.num_nodes for d in graph_data_list])} - {max([d.num_nodes for d in graph_data_list])}")
    print(f"Sample input count range: {min([d.input_count for d in graph_data_list])} - {max([d.input_count for d in graph_data_list])}")
    
    # Check feature structure matches GraphSAGE expectations
    print(f"\nFeature vector structure (first sample):")
    print(f"  Shape: {sample_data.x.shape}")
    print(f"  Data type: {sample_data.x.dtype}")
    print(f"  Edge index shape: {sample_data.edge_index.shape}")
    print(f"  Edge index data type: {sample_data.edge_index.dtype}")
    
    return class_counts

def save_converted_data(graph_data_list, output_filename='synthetic_circuit_data_classification.pkl'):
    """
    Save the converted data in the same format as the preprocessing pipeline
    """
    if not graph_data_list:
        print("No data to save")
        return
    
    # Save main data file
    with open(output_filename, 'wb') as f:
        pickle.dump(graph_data_list, f)
    
    # Create and save metadata
    class_counts = Counter([data.y.item() for data in graph_data_list])
    
    metadata = {
        'total_samples': len(graph_data_list),
        'num_classes': len(class_counts),
        'class_distribution': dict(class_counts),
        'circuits': list(set([data.circuit_name for data in graph_data_list])),
        'feature_dimensions': graph_data_list[0].x.shape[1],
        'sample_info': {
            'num_nodes_range': (min([data.num_nodes for data in graph_data_list]), 
                               max([data.num_nodes for data in graph_data_list])),
            'input_count_range': (min([data.input_count for data in graph_data_list]), 
                                 max([data.input_count for data in graph_data_list]))
        },
        'data_source': 'synthetic_boolean_circuits'
    }
    
    metadata_filename = output_filename.replace('.pkl', '_metadata.pkl')
    with open(metadata_filename, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved {len(graph_data_list)} processed data objects to '{output_filename}'")
    print(f"✓ Saved metadata to '{metadata_filename}'")
    print(f"✓ Feature dimensions: {metadata['feature_dimensions']}")
    print(f"✓ Number of classes: {metadata['num_classes']}")
    print("Synthetic data conversion complete!")

def main():
    """
    Main function to load synthetic data and convert to graph format
    """
    # Load synthetic dataset (you'll need to load your generated dataset)
    print("Loading synthetic dataset...")
    
    # This assumes you have saved your synthetic dataset as a pickle file
    # Adjust the filename as needed
    try:
        with open('synthetic_dataset.pkl', 'rb') as f:
            synthetic_dataset = pickle.load(f)
        print(f"Loaded {len(synthetic_dataset)} synthetic circuits")
    except FileNotFoundError:
        print("Error: synthetic_dataset.pkl not found. Please generate synthetic data first.")
        return
    
    # Convert to graph format
    graph_data_list = convert_synthetic_dataset_to_graph(synthetic_dataset)
    
    # Analyze converted data
    analyze_converted_data(graph_data_list)
    
    # Save converted data
    save_converted_data(graph_data_list)

if __name__ == "__main__":
    main()
