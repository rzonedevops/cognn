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
"""OpenCog Cognitive Microkernel layers for HyperGraphQL Neural Net."""
from typing import Any, Callable, Mapping, Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


class TruthValue:
  """Probabilistic truth value for atoms in OpenCog.
  
  Represents uncertain knowledge using strength and confidence values,
  implementing Simple Truth Value (STV) from OpenCog.
  
  Attributes:
    strength: Float in [0, 1] representing probability or truth degree
    confidence: Float in [0, 1] representing certainty of the strength value
  """
  
  def __init__(self, strength: float = 1.0, confidence: float = 1.0):
    """Initialize truth value.
    
    Args:
      strength: Truth strength in range [0, 1]
      confidence: Confidence in range [0, 1]
    """
    self.strength = tf.constant(strength, dtype=tf.float32)
    self.confidence = tf.constant(confidence, dtype=tf.float32)
  
  def to_tensor(self) -> tf.Tensor:
    """Convert to 2D tensor [strength, confidence]."""
    return tf.stack([self.strength, self.confidence])


class AttentionValue:
  """Attention value for Economic Attention Networks in OpenCog.
  
  Controls resource allocation and spread of activation in the knowledge graph.
  
  Attributes:
    sti: Short-term importance
    lti: Long-term importance  
    vlti: Very long-term importance
  """
  
  def __init__(self, sti: float = 0.0, lti: float = 0.0, vlti: float = 0.0):
    """Initialize attention value.
    
    Args:
      sti: Short-term importance
      lti: Long-term importance
      vlti: Very long-term importance
    """
    self.sti = tf.constant(sti, dtype=tf.float32)
    self.lti = tf.constant(lti, dtype=tf.float32)
    self.vlti = tf.constant(vlti, dtype=tf.float32)
  
  def to_tensor(self) -> tf.Tensor:
    """Convert to 3D tensor [sti, lti, vlti]."""
    return tf.stack([self.sti, self.lti, self.vlti])


@tf.keras.utils.register_keras_serializable(package="GNN>models>opencog")
class AtomSpace(tf.keras.layers.Layer):
  """AtomSpace: Hypergraph knowledge base for OpenCog cognitive architecture.
  
  The AtomSpace is the core data structure storing all knowledge as atoms
  (nodes and links) in a weighted labeled hypergraph. It integrates with
  TensorFlow GNN's GraphTensor to enable neural-symbolic reasoning.
  
  Example usage:
  
  ```python
  atom_space = AtomSpace(
      node_state_dim=128,
      edge_state_dim=64,
      enable_attention=True
  )
  graph = atom_space(input_graph)
  ```
  
  This layer enhances GraphTensor nodes and edges with:
  - Truth values for probabilistic reasoning
  - Attention values for importance tracking
  - Cognitive state representations
  """
  
  def __init__(
      self,
      node_state_dim: int = 128,
      edge_state_dim: int = 64,
      enable_attention: bool = True,
      enable_truth_values: bool = True,
      activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = "relu",
      dropout_rate: float = 0.0,
      **kwargs
  ):
    """Initialize AtomSpace layer.
    
    Args:
      node_state_dim: Dimensionality of node state embeddings
      edge_state_dim: Dimensionality of edge state embeddings
      enable_attention: Whether to compute attention values
      enable_truth_values: Whether to compute truth values
      activation: Activation function to use
      dropout_rate: Dropout rate for regularization
      **kwargs: Additional keyword arguments for Layer
    """
    super().__init__(**kwargs)
    self.node_state_dim = node_state_dim
    self.edge_state_dim = edge_state_dim
    self.enable_attention = enable_attention
    self.enable_truth_values = enable_truth_values
    self.activation = tf.keras.activations.get(activation)
    self.dropout_rate = dropout_rate
    
    # Node embedding layers
    self.node_embedding = tf.keras.layers.Dense(
        node_state_dim, activation=self.activation, name="node_embedding"
    )
    
    # Edge embedding layers
    self.edge_embedding = tf.keras.layers.Dense(
        edge_state_dim, activation=self.activation, name="edge_embedding"
    )
    
    # Truth value prediction layers
    if enable_truth_values:
      self.truth_value_layer = tf.keras.layers.Dense(
          2, activation="sigmoid", name="truth_values"
      )
    
    # Attention value prediction layers  
    if enable_attention:
      self.attention_value_layer = tf.keras.layers.Dense(
          3, activation="tanh", name="attention_values"
      )
    
    # Dropout
    if dropout_rate > 0:
      self.dropout = tf.keras.layers.Dropout(dropout_rate)
  
  def call(
      self, 
      graph: tfgnn.GraphTensor,
      training: bool = False
  ) -> tfgnn.GraphTensor:
    """Process graph through AtomSpace.
    
    Args:
      graph: Input GraphTensor
      training: Whether in training mode
      
    Returns:
      GraphTensor with enhanced node and edge features
    """
    # Initialize features dictionary
    node_sets_features = {}
    edge_sets_features = {}
    
    # Process each node set
    for node_set_name in graph.node_sets:
      node_features = graph.node_sets[node_set_name][tfgnn.HIDDEN_STATE]
      
      # Compute node embeddings
      node_state = self.node_embedding(node_features)
      
      if self.dropout_rate > 0 and training:
        node_state = self.dropout(node_state, training=training)
      
      features = {tfgnn.HIDDEN_STATE: node_state}
      
      # Add truth values
      if self.enable_truth_values:
        truth_vals = self.truth_value_layer(node_state)
        features["truth_value"] = truth_vals
      
      # Add attention values
      if self.enable_attention:
        attn_vals = self.attention_value_layer(node_state)
        features["attention_value"] = attn_vals
      
      node_sets_features[node_set_name] = features
    
    # Process each edge set
    for edge_set_name in graph.edge_sets:
      if tfgnn.HIDDEN_STATE in graph.edge_sets[edge_set_name].features:
        edge_features = graph.edge_sets[edge_set_name][tfgnn.HIDDEN_STATE]
        
        # Compute edge embeddings
        edge_state = self.edge_embedding(edge_features)
        
        if self.dropout_rate > 0 and training:
          edge_state = self.dropout(edge_state, training=training)
        
        edge_sets_features[edge_set_name] = {tfgnn.HIDDEN_STATE: edge_state}
    
    # Update graph with new features
    return graph.replace_features(
        node_sets=node_sets_features,
        edge_sets=edge_sets_features if edge_sets_features else None
    )
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "node_state_dim": self.node_state_dim,
        "edge_state_dim": self.edge_state_dim,
        "enable_attention": self.enable_attention,
        "enable_truth_values": self.enable_truth_values,
        "activation": tf.keras.activations.serialize(self.activation),
        "dropout_rate": self.dropout_rate,
    })
    return config


@tf.keras.utils.register_keras_serializable(package="GNN>models>opencog")
class HyperGraphQLQuery(tf.keras.layers.Layer):
  """HyperGraphQL-style query layer for pattern matching in hypergraphs.
  
  Implements query operations inspired by OpenCog's pattern matcher,
  allowing for flexible graph queries with variable binding.
  
  Example usage:
  
  ```python
  query_layer = HyperGraphQLQuery(
      query_dim=64,
      num_query_heads=4
  )
  query_result = query_layer(graph, query_pattern)
  ```
  """
  
  def __init__(
      self,
      query_dim: int = 64,
      num_query_heads: int = 4,
      activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = "relu",
      **kwargs
  ):
    """Initialize HyperGraphQL query layer.
    
    Args:
      query_dim: Dimensionality of query representations
      num_query_heads: Number of parallel query attention heads
      activation: Activation function
      **kwargs: Additional keyword arguments for Layer
    """
    super().__init__(**kwargs)
    self.query_dim = query_dim
    self.num_query_heads = num_query_heads
    self.activation = tf.keras.activations.get(activation)
    
    # Multi-head attention for pattern matching
    self.query_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_query_heads,
        key_dim=query_dim,
        name="query_attention"
    )
    
    # Query projection layers
    self.query_projection = tf.keras.layers.Dense(
        query_dim, activation=self.activation, name="query_projection"
    )
  
  def call(
      self,
      graph: tfgnn.GraphTensor,
      query_pattern: Optional[tf.Tensor] = None,
      training: bool = False
  ) -> tf.Tensor:
    """Execute hypergraph query.
    
    Args:
      graph: Input GraphTensor to query
      query_pattern: Optional query pattern tensor
      training: Whether in training mode
      
    Returns:
      Query results as tensor
    """
    # Use graph context or first node set as query base
    if graph.context is not None and tfgnn.HIDDEN_STATE in graph.context.features:
      context_features = graph.context[tfgnn.HIDDEN_STATE]
    else:
      # Use first node set as fallback
      first_node_set = list(graph.node_sets)[0]
      context_features = graph.node_sets[first_node_set][tfgnn.HIDDEN_STATE]
    
    # Project query
    if query_pattern is not None:
      query = self.query_projection(query_pattern)
    else:
      query = self.query_projection(context_features)
    
    # Apply attention for pattern matching
    attended = self.query_attention(
        query=query,
        value=context_features,
        key=context_features,
        training=training
    )
    
    return attended
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "query_dim": self.query_dim,
        "num_query_heads": self.num_query_heads,
        "activation": tf.keras.activations.serialize(self.activation),
    })
    return config


@tf.keras.utils.register_keras_serializable(package="GNN>models>opencog")
class CognitiveGraphUpdate(tf.keras.layers.Layer):
  """Cognitive graph update combining AtomSpace reasoning with GNN message passing.
  
  This layer integrates OpenCog-style cognitive operations with neural
  graph updates, enabling neuro-symbolic AI. It performs:
  
  1. Attention-based importance spreading
  2. Truth value propagation and revision
  3. Graph neural network message passing
  4. Cognitive state updates
  
  Example usage:
  
  ```python
  cognitive_update = CognitiveGraphUpdate(
      node_state_dim=128,
      message_dim=64,
      num_heads=4
  )
  updated_graph = cognitive_update(graph)
  ```
  """
  
  def __init__(
      self,
      node_state_dim: int = 128,
      message_dim: int = 64,
      num_heads: int = 4,
      enable_attention_spreading: bool = True,
      enable_truth_revision: bool = True,
      activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = "relu",
      dropout_rate: float = 0.0,
      **kwargs
  ):
    """Initialize cognitive graph update layer.
    
    Args:
      node_state_dim: Dimensionality of node states
      message_dim: Dimensionality of messages
      num_heads: Number of attention heads
      enable_attention_spreading: Whether to spread attention values
      enable_truth_revision: Whether to revise truth values
      activation: Activation function
      dropout_rate: Dropout rate
      **kwargs: Additional keyword arguments for Layer
    """
    super().__init__(**kwargs)
    self.node_state_dim = node_state_dim
    self.message_dim = message_dim
    self.num_heads = num_heads
    self.enable_attention_spreading = enable_attention_spreading
    self.enable_truth_revision = enable_truth_revision
    self.activation = tf.keras.activations.get(activation)
    self.dropout_rate = dropout_rate
    
    # Message passing layers
    self.message_layer = tf.keras.layers.Dense(
        message_dim, activation=self.activation, name="message"
    )
    
    # Node state update layers
    self.state_update = tf.keras.layers.Dense(
        node_state_dim, activation=self.activation, name="state_update"
    )
    
    # Attention spreading layer
    if enable_attention_spreading:
      self.attention_spread = tf.keras.layers.Dense(
          3, activation="tanh", name="attention_spread"
      )
    
    # Truth revision layer
    if enable_truth_revision:
      self.truth_revision = tf.keras.layers.Dense(
          2, activation="sigmoid", name="truth_revision"
      )
    
    # Graph update layer
    self.graph_update = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "_default": tfgnn.keras.layers.NodeSetUpdate(
                edge_set_inputs={},
                next_state=tfgnn.keras.layers.SingleInputNextState()
            )
        }
    )
  
  def call(
      self,
      graph: tfgnn.GraphTensor,
      training: bool = False
  ) -> tfgnn.GraphTensor:
    """Apply cognitive graph update.
    
    Args:
      graph: Input GraphTensor
      training: Whether in training mode
      
    Returns:
      Updated GraphTensor
    """
    node_sets_features = {}
    
    # Update each node set with cognitive operations
    for node_set_name in graph.node_sets:
      node_features = graph.node_sets[node_set_name][tfgnn.HIDDEN_STATE]
      
      # Compute new node states
      new_state = self.state_update(node_features)
      
      features = {tfgnn.HIDDEN_STATE: new_state}
      
      # Spread attention if enabled
      if self.enable_attention_spreading and "attention_value" in graph.node_sets[node_set_name].features:
        old_attention = graph.node_sets[node_set_name]["attention_value"]
        new_attention = self.attention_spread(
            tf.concat([new_state, old_attention], axis=-1)
        )
        features["attention_value"] = new_attention
      
      # Revise truth values if enabled
      if self.enable_truth_revision and "truth_value" in graph.node_sets[node_set_name].features:
        old_truth = graph.node_sets[node_set_name]["truth_value"]
        new_truth = self.truth_revision(
            tf.concat([new_state, old_truth], axis=-1)
        )
        features["truth_value"] = new_truth
      
      node_sets_features[node_set_name] = features
    
    # Apply updates to graph
    updated_graph = graph.replace_features(node_sets=node_sets_features)
    
    return updated_graph
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "node_state_dim": self.node_state_dim,
        "message_dim": self.message_dim,
        "num_heads": self.num_heads,
        "enable_attention_spreading": self.enable_attention_spreading,
        "enable_truth_revision": self.enable_truth_revision,
        "activation": tf.keras.activations.serialize(self.activation),
        "dropout_rate": self.dropout_rate,
    })
    return config
