#!/usr/bin/env python3
# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example demonstrating OpenCog cognitive microkernel with TensorFlow GNN.

This example shows how to use the OpenCog-inspired cognitive architecture
for neuro-symbolic AI on graph data. It demonstrates:

1. Creating a knowledge graph with semantic concepts
2. Initializing an AtomSpace with truth and attention values
3. Performing cognitive reasoning with graph updates
4. Querying the hypergraph with HyperGraphQL
"""

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog


def create_semantic_knowledge_graph():
  """Create a sample semantic knowledge graph.
  
  This creates a simple graph representing conceptual relationships:
  - Concepts as nodes (e.g., "cat", "animal", "mammal")
  - Relations as edges (e.g., "is-a", "has-property")
  
  Returns:
    A GraphTensor representing the knowledge graph.
  """
  # Create concept embeddings (one-hot encodings for simplicity)
  num_concepts = 5
  concept_features = tf.eye(num_concepts, dtype=tf.float32)
  
  # Create relation embeddings
  num_relations = 6
  relation_features = tf.random.uniform((num_relations, 4))
  
  # Define graph structure:
  # 0: cat, 1: dog, 2: mammal, 3: animal, 4: vertebrate
  # Relations: cat->mammal, dog->mammal, mammal->animal, 
  #            animal->vertebrate, cat->vertebrate, dog->vertebrate
  source_indices = [0, 1, 2, 3, 0, 1]
  target_indices = [2, 2, 3, 4, 4, 4]
  
  graph = tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(
          features={
              tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0, 0.0, 0.0]])
          }
      ),
      node_sets={
          "concepts": tfgnn.NodeSet.from_fields(
              sizes=[num_concepts],
              features={
                  tfgnn.HIDDEN_STATE: concept_features
              }
          )
      },
      edge_sets={
          "relations": tfgnn.EdgeSet.from_fields(
              sizes=[num_relations],
              adjacency=tfgnn.Adjacency.from_indices(
                  source=("concepts", source_indices),
                  target=("concepts", target_indices)
              ),
              features={
                  tfgnn.HIDDEN_STATE: relation_features
              }
          )
      }
  )
  
  return graph


def build_cognitive_reasoning_model(graph_spec):
  """Build a cognitive reasoning model.
  
  Args:
    graph_spec: GraphTensorSpec for the input graph.
    
  Returns:
    A Keras model that performs cognitive reasoning.
  """
  # Input layer
  graph = inputs = tf.keras.layers.Input(type_spec=graph_spec)
  
  # Initialize AtomSpace with truth and attention values
  print("Initializing AtomSpace...")
  graph = opencog.AtomSpace(
      node_state_dim=32,
      edge_state_dim=16,
      enable_attention=True,
      enable_truth_values=True,
      dropout_rate=0.1
  )(graph)
  
  # Perform multiple rounds of cognitive updates
  print("Applying cognitive graph updates...")
  for i in range(3):
    print(f"  Round {i+1}/3")
    graph = opencog.CognitiveGraphUpdate(
        node_state_dim=32,
        message_dim=16,
        num_heads=2,
        enable_attention_spreading=True,
        enable_truth_revision=True,
        dropout_rate=0.1
    )(graph)
  
  # Query the final graph state
  print("Creating query layer...")
  query_layer = opencog.HyperGraphQLQuery(
      query_dim=16,
      num_query_heads=2
  )
  output = query_layer(graph)
  
  # Build model
  model = tf.keras.Model(inputs, output)
  return model


def demonstrate_truth_and_attention_values():
  """Demonstrate truth values and attention values."""
  print("\n" + "="*60)
  print("Demonstrating Truth and Attention Values")
  print("="*60)
  
  # Create truth value
  tv = opencog.TruthValue(strength=0.85, confidence=0.90)
  print(f"\nTruth Value:")
  print(f"  Strength: {tv.strength.numpy():.2f}")
  print(f"  Confidence: {tv.confidence.numpy():.2f}")
  print(f"  As tensor: {tv.to_tensor().numpy()}")
  
  # Create attention value
  av = opencog.AttentionValue(sti=1.0, lti=0.5, vlti=0.1)
  print(f"\nAttention Value:")
  print(f"  STI (Short-term): {av.sti.numpy():.2f}")
  print(f"  LTI (Long-term): {av.lti.numpy():.2f}")
  print(f"  VLTI (Very long-term): {av.vlti.numpy():.2f}")
  print(f"  As tensor: {av.to_tensor().numpy()}")


def main():
  """Main demonstration function."""
  print("\n" + "="*60)
  print("OpenCog Cognitive Microkernel Demo")
  print("="*60)
  
  # Demonstrate truth and attention values
  demonstrate_truth_and_attention_values()
  
  # Create semantic knowledge graph
  print("\n" + "="*60)
  print("Creating Semantic Knowledge Graph")
  print("="*60)
  knowledge_graph = create_semantic_knowledge_graph()
  print(f"\nGraph structure:")
  print(f"  Concepts (nodes): {knowledge_graph.node_sets['concepts'].total_size}")
  print(f"  Relations (edges): {knowledge_graph.edge_sets['relations'].total_size}")
  
  # Build cognitive reasoning model
  print("\n" + "="*60)
  print("Building Cognitive Reasoning Model")
  print("="*60)
  graph_spec = knowledge_graph.spec
  model = build_cognitive_reasoning_model(graph_spec)
  
  # Print model summary
  print("\nModel Summary:")
  model.summary()
  
  # Run inference
  print("\n" + "="*60)
  print("Running Cognitive Inference")
  print("="*60)
  print("\nProcessing knowledge graph through cognitive layers...")
  output = model(knowledge_graph)
  print(f"\nOutput shape: {output.shape}")
  print(f"Output sample:\n{output.numpy()[:3]}")
  
  print("\n" + "="*60)
  print("Demo Complete!")
  print("="*60)
  print("\nThe cognitive microkernel has successfully:")
  print("  ✓ Initialized AtomSpace with truth and attention values")
  print("  ✓ Performed multi-round cognitive graph updates")
  print("  ✓ Spread attention and revised truth values")
  print("  ✓ Executed HyperGraphQL queries")
  print("\nThis demonstrates neuro-symbolic AI combining:")
  print("  • Neural learning (graph neural networks)")
  print("  • Symbolic reasoning (truth value propagation)")
  print("  • Attention management (economic attention networks)")
  print("  • Pattern matching (HyperGraphQL queries)")


if __name__ == "__main__":
  main()
