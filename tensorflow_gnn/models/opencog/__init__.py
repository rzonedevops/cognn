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
"""OpenCog Cognitive Microkernel with HyperGraphQL Neural Net.

This module implements OpenCog-inspired cognitive architecture using
TensorFlow GNN's HyperGraph capabilities. It provides:

- AtomSpace: A hypergraph knowledge base for storing atoms
- Atom types: Node and Link representations 
- Truth values and attention values for probabilistic reasoning
- HyperGraphQL-style query interface
- Integration with neural graph operations

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import opencog
```
"""

from tensorflow_gnn.models.opencog import layers
from tensorflow_gnn.utils import api_utils

# NOTE: This package is covered by tensorflow_gnn/api_def/api_symbols_test.py.
# Please see there for instructions how to reflect API changes.
# LINT.IfChange

AtomSpace = layers.AtomSpace
CognitiveGraphUpdate = layers.CognitiveGraphUpdate
HyperGraphQLQuery = layers.HyperGraphQLQuery
TruthValue = layers.TruthValue
AttentionValue = layers.AttentionValue

# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(__name__, [])
# LINT.ThenChange(../../api_def/opencog-symbols.txt)
