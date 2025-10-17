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
"""Tests for OpenCog cognitive microkernel layers."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog

# Enable graph tensor validation
tfgnn.enable_graph_tensor_validation_at_runtime()


class TruthValueTest(tf.test.TestCase):
  """Tests for TruthValue class."""

  def test_initialization(self):
    """Test truth value initialization."""
    tv = opencog.TruthValue(strength=0.8, confidence=0.9)
    self.assertAlmostEqual(tv.strength.numpy(), 0.8, places=5)
    self.assertAlmostEqual(tv.confidence.numpy(), 0.9, places=5)

  def test_to_tensor(self):
    """Test conversion to tensor."""
    tv = opencog.TruthValue(strength=0.7, confidence=0.6)
    tensor = tv.to_tensor()
    self.assertEqual(tensor.shape, (2,))
    self.assertAlmostEqual(tensor[0].numpy(), 0.7, places=5)
    self.assertAlmostEqual(tensor[1].numpy(), 0.6, places=5)


class AttentionValueTest(tf.test.TestCase):
  """Tests for AttentionValue class."""

  def test_initialization(self):
    """Test attention value initialization."""
    av = opencog.AttentionValue(sti=1.0, lti=0.5, vlti=0.1)
    self.assertAlmostEqual(av.sti.numpy(), 1.0, places=5)
    self.assertAlmostEqual(av.lti.numpy(), 0.5, places=5)
    self.assertAlmostEqual(av.vlti.numpy(), 0.1, places=5)

  def test_to_tensor(self):
    """Test conversion to tensor."""
    av = opencog.AttentionValue(sti=0.8, lti=0.4, vlti=0.2)
    tensor = av.to_tensor()
    self.assertEqual(tensor.shape, (3,))
    self.assertAlmostEqual(tensor[0].numpy(), 0.8, places=5)
    self.assertAlmostEqual(tensor[1].numpy(), 0.4, places=5)
    self.assertAlmostEqual(tensor[2].numpy(), 0.2, places=5)


class AtomSpaceTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for AtomSpace layer."""

  def _create_test_graph(self):
    """Create a simple test graph."""
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=[3],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                    )
                }
            )
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("nodes", [0, 1]),
                    target=("nodes", [1, 2])
                ),
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[0.1, 0.2], [0.3, 0.4]])
                }
            )
        }
    )

  def test_atom_space_basic(self):
    """Test basic AtomSpace functionality."""
    graph = self._create_test_graph()
    atom_space = opencog.AtomSpace(
        node_state_dim=8,
        edge_state_dim=4
    )
    
    output_graph = atom_space(graph)
    
    # Check output shape
    self.assertEqual(
        output_graph.node_sets["nodes"][tfgnn.HIDDEN_STATE].shape,
        (3, 8)
    )
    self.assertEqual(
        output_graph.edge_sets["edges"][tfgnn.HIDDEN_STATE].shape,
        (2, 4)
    )

  def test_atom_space_with_truth_values(self):
    """Test AtomSpace with truth values enabled."""
    graph = self._create_test_graph()
    atom_space = opencog.AtomSpace(
        node_state_dim=8,
        enable_truth_values=True,
        enable_attention=False
    )
    
    output_graph = atom_space(graph)
    
    # Check that truth values are added
    self.assertIn("truth_value", output_graph.node_sets["nodes"].features)
    self.assertEqual(
        output_graph.node_sets["nodes"]["truth_value"].shape,
        (3, 2)
    )

  def test_atom_space_with_attention_values(self):
    """Test AtomSpace with attention values enabled."""
    graph = self._create_test_graph()
    atom_space = opencog.AtomSpace(
        node_state_dim=8,
        enable_truth_values=False,
        enable_attention=True
    )
    
    output_graph = atom_space(graph)
    
    # Check that attention values are added
    self.assertIn("attention_value", output_graph.node_sets["nodes"].features)
    self.assertEqual(
        output_graph.node_sets["nodes"]["attention_value"].shape,
        (3, 3)
    )

  def test_atom_space_dropout(self):
    """Test AtomSpace with dropout."""
    graph = self._create_test_graph()
    atom_space = opencog.AtomSpace(
        node_state_dim=8,
        dropout_rate=0.5
    )
    
    # Should work in both training and inference modes
    output_train = atom_space(graph, training=True)
    output_infer = atom_space(graph, training=False)
    
    self.assertEqual(output_train.node_sets["nodes"][tfgnn.HIDDEN_STATE].shape, (3, 8))
    self.assertEqual(output_infer.node_sets["nodes"][tfgnn.HIDDEN_STATE].shape, (3, 8))

  def test_atom_space_serialization(self):
    """Test that AtomSpace can be serialized and deserialized."""
    atom_space = opencog.AtomSpace(
        node_state_dim=16,
        edge_state_dim=8,
        enable_attention=True,
        enable_truth_values=True,
        dropout_rate=0.1
    )
    
    config = atom_space.get_config()
    recreated = opencog.AtomSpace.from_config(config)
    
    self.assertEqual(recreated.node_state_dim, 16)
    self.assertEqual(recreated.edge_state_dim, 8)
    self.assertTrue(recreated.enable_attention)
    self.assertTrue(recreated.enable_truth_values)
    self.assertAlmostEqual(recreated.dropout_rate, 0.1, places=5)


class HyperGraphQLQueryTest(tf.test.TestCase):
  """Tests for HyperGraphQLQuery layer."""

  def _create_test_graph(self):
    """Create a test graph with context."""
    return tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={
                tfgnn.HIDDEN_STATE: tf.constant([[1.0, 2.0, 3.0, 4.0]])
            }
        ),
        node_sets={
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=[3],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]]
                    )
                }
            )
        }
    )

  def test_query_basic(self):
    """Test basic query functionality."""
    graph = self._create_test_graph()
    query_layer = opencog.HyperGraphQLQuery(
        query_dim=4,
        num_query_heads=2
    )
    
    result = query_layer(graph)
    
    # Result should have proper shape
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(result.shape[-1], 4)

  def test_query_with_pattern(self):
    """Test query with explicit pattern."""
    graph = self._create_test_graph()
    query_layer = opencog.HyperGraphQLQuery(
        query_dim=4,
        num_query_heads=2
    )
    
    query_pattern = tf.constant([[2.0, 3.0, 4.0, 5.0]])
    result = query_layer(graph, query_pattern=query_pattern)
    
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(result.shape[-1], 4)

  def test_query_serialization(self):
    """Test query layer serialization."""
    query_layer = opencog.HyperGraphQLQuery(
        query_dim=8,
        num_query_heads=4
    )
    
    config = query_layer.get_config()
    recreated = opencog.HyperGraphQLQuery.from_config(config)
    
    self.assertEqual(recreated.query_dim, 8)
    self.assertEqual(recreated.num_query_heads, 4)


class CognitiveGraphUpdateTest(tf.test.TestCase):
  """Tests for CognitiveGraphUpdate layer."""

  def _create_test_graph_with_values(self):
    """Create a test graph with truth and attention values."""
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=[3],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                    ),
                    "truth_value": tf.constant(
                        [[0.8, 0.9], [0.7, 0.8], [0.6, 0.7]]
                    ),
                    "attention_value": tf.constant(
                        [[1.0, 0.5, 0.1], [0.8, 0.4, 0.2], [0.6, 0.3, 0.1]]
                    )
                }
            )
        }
    )

  def test_cognitive_update_basic(self):
    """Test basic cognitive update."""
    graph = self._create_test_graph_with_values()
    update_layer = opencog.CognitiveGraphUpdate(
        node_state_dim=8,
        message_dim=4
    )
    
    updated_graph = update_layer(graph)
    
    # Check output shape
    self.assertEqual(
        updated_graph.node_sets["nodes"][tfgnn.HIDDEN_STATE].shape,
        (3, 8)
    )

  def test_cognitive_update_with_attention_spreading(self):
    """Test cognitive update with attention spreading."""
    graph = self._create_test_graph_with_values()
    update_layer = opencog.CognitiveGraphUpdate(
        node_state_dim=8,
        enable_attention_spreading=True,
        enable_truth_revision=False
    )
    
    updated_graph = update_layer(graph)
    
    # Check that attention values are updated
    self.assertIn("attention_value", updated_graph.node_sets["nodes"].features)
    self.assertEqual(
        updated_graph.node_sets["nodes"]["attention_value"].shape,
        (3, 3)
    )

  def test_cognitive_update_with_truth_revision(self):
    """Test cognitive update with truth revision."""
    graph = self._create_test_graph_with_values()
    update_layer = opencog.CognitiveGraphUpdate(
        node_state_dim=8,
        enable_attention_spreading=False,
        enable_truth_revision=True
    )
    
    updated_graph = update_layer(graph)
    
    # Check that truth values are updated
    self.assertIn("truth_value", updated_graph.node_sets["nodes"].features)
    self.assertEqual(
        updated_graph.node_sets["nodes"]["truth_value"].shape,
        (3, 2)
    )

  def test_cognitive_update_serialization(self):
    """Test cognitive update serialization."""
    update_layer = opencog.CognitiveGraphUpdate(
        node_state_dim=16,
        message_dim=8,
        num_heads=4,
        enable_attention_spreading=True,
        enable_truth_revision=True
    )
    
    config = update_layer.get_config()
    recreated = opencog.CognitiveGraphUpdate.from_config(config)
    
    self.assertEqual(recreated.node_state_dim, 16)
    self.assertEqual(recreated.message_dim, 8)
    self.assertEqual(recreated.num_heads, 4)
    self.assertTrue(recreated.enable_attention_spreading)
    self.assertTrue(recreated.enable_truth_revision)


if __name__ == "__main__":
  tf.test.main()
