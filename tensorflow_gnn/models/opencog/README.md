# OpenCog Cognitive Microkernel for TensorFlow GNN

This module implements an OpenCog-inspired cognitive architecture using TensorFlow GNN's HyperGraph capabilities, creating a cognitive microkernel-based HyperGraphQL Neural Net.

## Overview

OpenCog is a cognitive architecture framework that uses a weighted labeled hypergraph (AtomSpace) as its core knowledge representation. This implementation integrates OpenCog concepts with TensorFlow GNN to enable neuro-symbolic AI:

- **AtomSpace**: Hypergraph knowledge base for storing atoms (nodes and links)
- **Truth Values**: Probabilistic reasoning with strength and confidence
- **Attention Values**: Economic attention networks for resource allocation
- **HyperGraphQL**: Pattern matching and query interface for hypergraphs
- **Cognitive Graph Updates**: Neural message passing with symbolic reasoning

## Key Components

### 1. AtomSpace Layer

The `AtomSpace` layer is the core component that enhances graph tensors with cognitive features:

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog

# Create an AtomSpace layer
atom_space = opencog.AtomSpace(
    node_state_dim=128,
    edge_state_dim=64,
    enable_attention=True,
    enable_truth_values=True,
    dropout_rate=0.1
)

# Apply to a graph
enhanced_graph = atom_space(input_graph)
```

Features:
- Embeds nodes and edges into cognitive state representations
- Computes truth values (strength, confidence) for probabilistic reasoning
- Tracks attention values (STI, LTI, VLTI) for importance-based processing
- Integrates with standard GNN operations

### 2. Truth Values

Truth values represent uncertain knowledge using Simple Truth Value (STV) semantics:

```python
# Create a truth value
tv = opencog.TruthValue(strength=0.8, confidence=0.9)

# Convert to tensor for neural processing
tv_tensor = tv.to_tensor()  # Shape: [2] -> [strength, confidence]
```

- **Strength**: Probability or truth degree in [0, 1]
- **Confidence**: Certainty of the strength value in [0, 1]

### 3. Attention Values

Attention values control spreading activation and resource allocation:

```python
# Create an attention value
av = opencog.AttentionValue(sti=1.0, lti=0.5, vlti=0.1)

# Convert to tensor
av_tensor = av.to_tensor()  # Shape: [3] -> [STI, LTI, VLTI]
```

- **STI** (Short-term importance): Current focus of attention
- **LTI** (Long-term importance): Long-term significance
- **VLTI** (Very long-term importance): Permanent importance

### 4. HyperGraphQL Query Layer

Pattern matching and query operations on hypergraphs:

```python
# Create a query layer
query_layer = opencog.HyperGraphQLQuery(
    query_dim=64,
    num_query_heads=4
)

# Execute a query
results = query_layer(graph, query_pattern=pattern_tensor)
```

Features:
- Multi-head attention for pattern matching
- Flexible variable binding
- Subgraph retrieval

### 5. Cognitive Graph Update

Combines neural graph updates with cognitive reasoning:

```python
# Create cognitive update layer
cognitive_update = opencog.CognitiveGraphUpdate(
    node_state_dim=128,
    message_dim=64,
    num_heads=4,
    enable_attention_spreading=True,
    enable_truth_revision=True
)

# Apply cognitive update
updated_graph = cognitive_update(graph)
```

Features:
- Attention spreading across the graph
- Truth value revision and propagation
- Integration with GNN message passing
- Cognitive state evolution

## Example Usage

### Complete Cognitive Neural Network

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog

# Define model
def build_cognitive_model(graph_spec):
    # Input
    graph = inputs = tf.keras.layers.Input(type_spec=graph_spec)
    
    # Initialize AtomSpace
    graph = opencog.AtomSpace(
        node_state_dim=128,
        edge_state_dim=64,
        enable_attention=True,
        enable_truth_values=True
    )(graph)
    
    # Multiple rounds of cognitive updates
    for _ in range(4):
        graph = opencog.CognitiveGraphUpdate(
            node_state_dim=128,
            message_dim=64,
            num_heads=4,
            enable_attention_spreading=True,
            enable_truth_revision=True
        )(graph)
    
    # Query for final output
    query_layer = opencog.HyperGraphQLQuery(
        query_dim=64,
        num_query_heads=4
    )
    output = query_layer(graph)
    
    return tf.keras.Model(inputs, output)

# Build and compile
model = build_cognitive_model(graph_tensor_spec)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### Neuro-Symbolic Reasoning

```python
# Create a graph with semantic knowledge
graph = tfgnn.GraphTensor.from_pieces(
    node_sets={
        "concepts": tfgnn.NodeSet.from_fields(
            sizes=[10],
            features={
                tfgnn.HIDDEN_STATE: concept_embeddings
            }
        )
    },
    edge_sets={
        "relations": tfgnn.EdgeSet.from_fields(
            sizes=[20],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("concepts", sources),
                target=("concepts", targets)
            ),
            features={
                tfgnn.HIDDEN_STATE: relation_embeddings
            }
        )
    }
)

# Apply cognitive reasoning
atom_space = opencog.AtomSpace(
    node_state_dim=256,
    enable_truth_values=True
)
reasoning_graph = atom_space(graph)

# Access truth values for probabilistic inference
concept_truth_values = reasoning_graph.node_sets["concepts"]["truth_value"]
```

## Architecture

The cognitive microkernel architecture follows this flow:

```
Input GraphTensor
    ↓
AtomSpace Initialization
    ↓ (adds truth_value, attention_value features)
Cognitive Graph Updates (×N rounds)
    ↓ (neural message passing + symbolic reasoning)
    ├─ Attention Spreading
    ├─ Truth Value Revision
    └─ State Evolution
    ↓
HyperGraphQL Query
    ↓ (pattern matching and retrieval)
Output
```

## Integration with OpenCog Concepts

This implementation maps OpenCog concepts to TensorFlow GNN:

| OpenCog Concept | TF-GNN Implementation |
|----------------|----------------------|
| Atom | GraphTensor node or edge |
| AtomSpace | Enhanced GraphTensor with cognitive features |
| Truth Value | 2D feature tensor [strength, confidence] |
| Attention Value | 3D feature tensor [STI, LTI, VLTI] |
| Pattern Matcher | HyperGraphQLQuery layer |
| PLN (reasoning) | Truth value propagation in updates |
| ECAN (attention) | Attention spreading in updates |

## Benefits

1. **Neuro-Symbolic AI**: Combines neural learning with symbolic reasoning
2. **Uncertainty Handling**: Truth values represent probabilistic knowledge
3. **Attention Management**: Focus computational resources on important atoms
4. **Pattern Matching**: Flexible queries on hypergraph structures
5. **Scalability**: Leverages TensorFlow's distributed computation
6. **Integration**: Works seamlessly with existing TF-GNN models

## References

- OpenCog Framework: https://opencog.org/
- OpenCog AtomSpace: https://wiki.opencog.org/w/AtomSpace
- Probabilistic Logic Networks (PLN): https://wiki.opencog.org/w/PLN
- Economic Attention Networks (ECAN): https://wiki.opencog.org/w/ECAN
- TensorFlow GNN: https://github.com/tensorflow/gnn

## Citation

If you use this cognitive microkernel implementation, please cite both the TF-GNN library and OpenCog project.
