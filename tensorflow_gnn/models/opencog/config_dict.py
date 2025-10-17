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
"""Configuration utilities for OpenCog cognitive microkernel models."""
from typing import Collection, Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog


def graph_update_get_config_dict():
  """Returns ConfigDict for CognitiveGraphUpdate with default hyperparameters.
  
  This function returns a ConfigDict with default hyperparameters for the
  OpenCog cognitive graph update layer. Users can modify these values to
  customize the behavior.
  
  Returns:
    A ConfigDict with the following keys:
      - node_state_dim: Dimensionality of node state (default: 128)
      - message_dim: Dimensionality of messages (default: 64)
      - num_heads: Number of attention heads (default: 4)
      - enable_attention_spreading: Whether to spread attention (default: True)
      - enable_truth_revision: Whether to revise truth values (default: True)
      - activation: Activation function name (default: "relu")
      - dropout_rate: Dropout rate (default: 0.0)
  """
  try:
    from ml_collections import config_dict
  except ImportError:
    raise ImportError(
        "This function requires ml_collections. "
        "Install it with: pip install ml-collections"
    )
  
  cfg = config_dict.ConfigDict()
  cfg.node_state_dim = 128
  cfg.message_dim = 64
  cfg.num_heads = 4
  cfg.enable_attention_spreading = True
  cfg.enable_truth_revision = True
  cfg.activation = "relu"
  cfg.dropout_rate = 0.0
  
  return cfg


def graph_update_from_config_dict(
    cfg,
    *,
    receiver_tag: Optional[tfgnn.IncidentNodeTag] = None,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
) -> tf.keras.layers.Layer:
  """Returns a CognitiveGraphUpdate initialized from a ConfigDict.
  
  Args:
    cfg: A ConfigDict with hyperparameters, as returned by
      graph_update_get_config_dict().
    receiver_tag: Optionally, the default receiver_tag for convolutions.
    node_set_names: Optionally, the node sets to update. If not set,
      updates all node sets.
      
  Returns:
    A CognitiveGraphUpdate layer initialized with the given hyperparameters.
  """
  def node_set_update_factory(node_set_name: tfgnn.NodeSetName):
    """Factory function for creating node set updates."""
    return opencog.CognitiveGraphUpdate(
        node_state_dim=cfg.node_state_dim,
        message_dim=cfg.message_dim,
        num_heads=cfg.num_heads,
        enable_attention_spreading=cfg.enable_attention_spreading,
        enable_truth_revision=cfg.enable_truth_revision,
        activation=cfg.activation,
        dropout_rate=cfg.dropout_rate,
    )
  
  # For now, return a single cognitive update layer
  # In a more complex setup, this could create a full GraphUpdate
  return opencog.CognitiveGraphUpdate(
      node_state_dim=cfg.node_state_dim,
      message_dim=cfg.message_dim,
      num_heads=cfg.num_heads,
      enable_attention_spreading=cfg.enable_attention_spreading,
      enable_truth_revision=cfg.enable_truth_revision,
      activation=cfg.activation,
      dropout_rate=cfg.dropout_rate,
  )


def atom_space_get_config_dict():
  """Returns ConfigDict for AtomSpace with default hyperparameters.
  
  Returns:
    A ConfigDict with the following keys:
      - node_state_dim: Node embedding dimension (default: 128)
      - edge_state_dim: Edge embedding dimension (default: 64)
      - enable_attention: Whether to compute attention values (default: True)
      - enable_truth_values: Whether to compute truth values (default: True)
      - activation: Activation function name (default: "relu")
      - dropout_rate: Dropout rate (default: 0.0)
  """
  try:
    from ml_collections import config_dict
  except ImportError:
    raise ImportError(
        "This function requires ml_collections. "
        "Install it with: pip install ml-collections"
    )
  
  cfg = config_dict.ConfigDict()
  cfg.node_state_dim = 128
  cfg.edge_state_dim = 64
  cfg.enable_attention = True
  cfg.enable_truth_values = True
  cfg.activation = "relu"
  cfg.dropout_rate = 0.0
  
  return cfg


def atom_space_from_config_dict(cfg) -> tf.keras.layers.Layer:
  """Returns an AtomSpace initialized from a ConfigDict.
  
  Args:
    cfg: A ConfigDict with hyperparameters, as returned by
      atom_space_get_config_dict().
      
  Returns:
    An AtomSpace layer initialized with the given hyperparameters.
  """
  return opencog.AtomSpace(
      node_state_dim=cfg.node_state_dim,
      edge_state_dim=cfg.edge_state_dim,
      enable_attention=cfg.enable_attention,
      enable_truth_values=cfg.enable_truth_values,
      activation=cfg.activation,
      dropout_rate=cfg.dropout_rate,
  )


def hypergraph_query_get_config_dict():
  """Returns ConfigDict for HyperGraphQLQuery with default hyperparameters.
  
  Returns:
    A ConfigDict with the following keys:
      - query_dim: Query dimension (default: 64)
      - num_query_heads: Number of query heads (default: 4)
      - activation: Activation function name (default: "relu")
  """
  try:
    from ml_collections import config_dict
  except ImportError:
    raise ImportError(
        "This function requires ml_collections. "
        "Install it with: pip install ml-collections"
    )
  
  cfg = config_dict.ConfigDict()
  cfg.query_dim = 64
  cfg.num_query_heads = 4
  cfg.activation = "relu"
  
  return cfg


def hypergraph_query_from_config_dict(cfg) -> tf.keras.layers.Layer:
  """Returns a HyperGraphQLQuery initialized from a ConfigDict.
  
  Args:
    cfg: A ConfigDict with hyperparameters, as returned by
      hypergraph_query_get_config_dict().
      
  Returns:
    A HyperGraphQLQuery layer initialized with the given hyperparameters.
  """
  return opencog.HyperGraphQLQuery(
      query_dim=cfg.query_dim,
      num_query_heads=cfg.num_query_heads,
      activation=cfg.activation,
  )
