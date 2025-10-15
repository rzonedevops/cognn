# OpenCog Cognitive Microkernel Implementation

## Overview

This document describes the implementation of an OpenCog-inspired cognitive microkernel using TensorFlow GNN's HyperGraph capabilities, creating a cognitive microkernel-based HyperGraphQL Neural Net.

## Problem Statement

**Goal**: Implement OpenCog as a cognitive microkernel-based HyperGraphQL Neural Net

**Solution**: Created a complete neuro-symbolic AI framework that integrates OpenCog's cognitive architecture concepts with TensorFlow GNN, enabling probabilistic reasoning, attention management, and pattern matching on hypergraphs.

## Architecture

### Core Components

#### 1. AtomSpace Layer (`AtomSpace`)
The central hypergraph knowledge base that stores and processes atoms (nodes and links).

**Features**:
- Embeds nodes and edges into cognitive state representations
- Computes truth values (strength, confidence) for probabilistic reasoning
- Tracks attention values (STI, LTI, VLTI) for importance-based processing
- Integrates seamlessly with standard GNN operations
- Supports dropout regularization

**Parameters**:
- `node_state_dim`: Dimensionality of node embeddings (default: 128)
- `edge_state_dim`: Dimensionality of edge embeddings (default: 64)
- `enable_attention`: Compute attention values (default: True)
- `enable_truth_values`: Compute truth values (default: True)
- `activation`: Activation function (default: "relu")
- `dropout_rate`: Dropout rate (default: 0.0)

#### 2. Truth Values (`TruthValue`)
Represents uncertain knowledge using Simple Truth Value (STV) semantics from OpenCog.

**Components**:
- `strength`: Probability or truth degree in [0, 1]
- `confidence`: Certainty of the strength value in [0, 1]

**Usage**: Enables probabilistic logic networks (PLN) style reasoning.

#### 3. Attention Values (`AttentionValue`)
Controls spreading activation and resource allocation (Economic Attention Networks - ECAN).

**Components**:
- `sti`: Short-term importance (current focus)
- `lti`: Long-term importance (long-term significance)
- `vlti`: Very long-term importance (permanent importance)

**Usage**: Manages computational resources by focusing on important atoms.

#### 4. HyperGraphQL Query Layer (`HyperGraphQLQuery`)
Pattern matching and query operations inspired by OpenCog's pattern matcher.

**Features**:
- Multi-head attention for flexible pattern matching
- Variable binding in graph queries
- Subgraph retrieval

**Parameters**:
- `query_dim`: Dimensionality of query representations (default: 64)
- `num_query_heads`: Number of parallel query attention heads (default: 4)
- `activation`: Activation function (default: "relu")

#### 5. Cognitive Graph Update Layer (`CognitiveGraphUpdate`)
Combines neural graph updates with cognitive reasoning operations.

**Operations**:
1. Neural message passing between nodes
2. Attention spreading across the graph
3. Truth value revision and propagation
4. Cognitive state evolution

**Parameters**:
- `node_state_dim`: Dimensionality of node states (default: 128)
- `message_dim`: Dimensionality of messages (default: 64)
- `num_heads`: Number of attention heads (default: 4)
- `enable_attention_spreading`: Spread attention values (default: True)
- `enable_truth_revision`: Revise truth values (default: True)
- `activation`: Activation function (default: "relu")
- `dropout_rate`: Dropout rate (default: 0.0)

## Implementation Details

### File Structure

```
tensorflow_gnn/models/opencog/
├── __init__.py              # Module initialization and API exports (57 lines)
├── layers.py                # Core layer implementations (477 lines)
├── layers_test.py           # Comprehensive unit tests (349 lines)
├── config_dict.py           # Configuration utilities (193 lines)
├── BUILD                    # Bazel build configuration (63 lines)
└── README.md                # Documentation and examples (256 lines)

examples/
└── opencog_cognitive_example.py  # Runnable demonstration (212 lines)
```

**Total**: ~1,607 lines of code and documentation

### Integration with TensorFlow GNN

The implementation leverages TensorFlow GNN's core components:

- **GraphTensor**: Base structure for representing hypergraphs
- **HyperAdjacency**: Support for hyperedges connecting multiple nodes
- **NodeSet/EdgeSet**: Collections of nodes and edges with features
- **GraphUpdate**: Framework for iterative graph updates
- **Keras Layers**: Full integration with TensorFlow's Keras API

### OpenCog Concept Mappings

| OpenCog Concept | TF-GNN Implementation |
|----------------|----------------------|
| Atom | GraphTensor node or edge |
| AtomSpace | Enhanced GraphTensor with cognitive features |
| Node Atom | NodeSet entry |
| Link Atom | EdgeSet entry (can be hyperedge) |
| Truth Value | 2D feature tensor [strength, confidence] |
| Attention Value | 3D feature tensor [STI, LTI, VLTI] |
| Pattern Matcher | HyperGraphQLQuery layer |
| PLN (Probabilistic Logic Networks) | Truth value propagation in updates |
| ECAN (Economic Attention Networks) | Attention spreading in updates |

## Usage Examples

### Basic Example

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog

# Initialize AtomSpace
atom_space = opencog.AtomSpace(
    node_state_dim=128,
    enable_attention=True,
    enable_truth_values=True
)

# Apply to graph
enhanced_graph = atom_space(input_graph)

# Access cognitive features
truth_values = enhanced_graph.node_sets["concepts"]["truth_value"]
attention_values = enhanced_graph.node_sets["concepts"]["attention_value"]
```

### Cognitive Reasoning Pipeline

```python
# Build complete cognitive reasoning model
def build_cognitive_model(graph_spec):
    graph = inputs = tf.keras.layers.Input(type_spec=graph_spec)
    
    # Initialize AtomSpace
    graph = opencog.AtomSpace(
        node_state_dim=128,
        edge_state_dim=64
    )(graph)
    
    # Multiple rounds of cognitive updates
    for _ in range(4):
        graph = opencog.CognitiveGraphUpdate(
            node_state_dim=128,
            message_dim=64,
            enable_attention_spreading=True,
            enable_truth_revision=True
        )(graph)
    
    # Query for results
    query_layer = opencog.HyperGraphQLQuery(query_dim=64)
    output = query_layer(graph)
    
    return tf.keras.Model(inputs, output)
```

### Using Configuration Utilities

```python
from tensorflow_gnn.models.opencog import config_dict

# Get default configuration
cfg = config_dict.graph_update_get_config_dict()

# Customize hyperparameters
cfg.node_state_dim = 256
cfg.num_heads = 8
cfg.dropout_rate = 0.1

# Create layer from config
layer = config_dict.graph_update_from_config_dict(cfg)
```

## Testing

Comprehensive unit tests cover:

- Truth value creation and tensor conversion
- Attention value creation and tensor conversion
- AtomSpace initialization and graph processing
- Truth value computation
- Attention value computation
- Dropout behavior
- Layer serialization/deserialization
- HyperGraphQL query operations
- Pattern matching with explicit query patterns
- Cognitive graph updates
- Attention spreading
- Truth value revision
- Configuration utilities

Run tests with:
```bash
bazel test //tensorflow_gnn/models/opencog:layers_test
```

## Example Application

See `examples/opencog_cognitive_example.py` for a complete working example that demonstrates:

1. Creating a semantic knowledge graph
2. Initializing AtomSpace with truth and attention values
3. Performing multi-round cognitive updates
4. Querying the hypergraph
5. Extracting cognitive features

Run the example:
```bash
python3 examples/opencog_cognitive_example.py
```

## Benefits of This Implementation

1. **Neuro-Symbolic AI**: Combines neural learning (GNNs) with symbolic reasoning (truth values, logic)
2. **Uncertainty Handling**: Truth values represent and propagate probabilistic knowledge
3. **Attention Management**: Economic attention networks focus resources on important information
4. **Pattern Matching**: Flexible HyperGraphQL queries for complex graph patterns
5. **Scalability**: Leverages TensorFlow's distributed computation and GPU acceleration
6. **Integration**: Works seamlessly with existing TF-GNN models and layers
7. **Extensibility**: Clear API for adding new cognitive operations
8. **Production-Ready**: Full Keras support for training, inference, and deployment

## Differences from Traditional OpenCog

While inspired by OpenCog, this implementation adapts the concepts for neural computation:

1. **Continuous vs Discrete**: Uses continuous embeddings instead of discrete symbolic atoms
2. **Neural PLN**: Truth value propagation through neural networks instead of logical inference rules
3. **Differentiable**: All operations are differentiable for end-to-end training
4. **Batched**: Processes batches of graphs efficiently using TensorFlow
5. **Distributed**: Can leverage TensorFlow's distributed training capabilities

## Future Extensions

Possible enhancements:

1. **Advanced Truth Value Types**: Implement other TV types (IndefiniteTV, DistributionalTV)
2. **More PLN Rules**: Add specific probabilistic inference rules
3. **Cognitive Schemata**: Implement procedural knowledge representation
4. **Attention Allocation**: More sophisticated ECAN algorithms
5. **Pattern Matcher Enhancements**: Variable types, constraints, and complex patterns
6. **Integration with Other Cognitive Operations**: Planning, goal management, etc.

## References

- **OpenCog Framework**: https://opencog.org/
- **OpenCog AtomSpace**: https://wiki.opencog.org/w/AtomSpace
- **Probabilistic Logic Networks (PLN)**: https://wiki.opencog.org/w/PLN
- **Economic Attention Networks (ECAN)**: https://wiki.opencog.org/w/ECAN
- **TensorFlow GNN**: https://github.com/tensorflow/gnn
- **HyperGraphs**: https://en.wikipedia.org/wiki/Hypergraph

## Citation

When using this implementation, please cite both TensorFlow GNN and the OpenCog project:

```bibtex
@article{tfgnn,
  title={{TF-GNN:} Graph Neural Networks in TensorFlow},
  author={Ferludin, Oleksandr and others},
  journal={CoRR},
  volume={abs/2207.03522},
  year={2023}
}

@misc{opencog,
  title={The OpenCog Framework},
  howpublished={\url{https://opencog.org/}},
  note={Accessed: 2025-10-15}
}
```

## License

This implementation is licensed under the Apache License 2.0, consistent with TensorFlow GNN.

## Contributing

Contributions are welcome! Please follow the TensorFlow GNN contribution guidelines and ensure:
- All tests pass
- Code follows the existing style
- Documentation is updated
- New features include tests

---

**Implementation completed**: October 15, 2025  
**Total lines of code**: ~1,607 (including tests and documentation)  
**Status**: Ready for use and further development
