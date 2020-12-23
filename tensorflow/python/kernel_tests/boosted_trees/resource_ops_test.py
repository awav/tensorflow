# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for boosted_trees resource kernels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
_p = print
_p("IMPORTING", file=sys.stderr)

from google.protobuf import text_format
_p("IMPORTING 2", file=sys.stderr)

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
_p("IMPORTING 3", file=sys.stderr)
from tensorflow.python.framework import ops
_p("IMPORTING 4", file=sys.stderr)
from tensorflow.python.framework import test_util
_p("IMPORTING 5", file=sys.stderr)
from tensorflow.python.ops import boosted_trees_ops
_p("IMPORTING 6", file=sys.stderr)
from tensorflow.python.ops import resources
_p("IMPORTING 7", file=sys.stderr)
from tensorflow.python.platform import googletest
_p("IMPORTING 8", file=sys.stderr)


class ResourceOpsTest(test_util.TensorFlowTestCase):
  """Tests resource_ops."""

  @test_util.run_deprecated_v1
  def testCreate(self):
    _p("testCreate 1", file=sys.stderr)
    with self.cached_session():
      _p("testCreate 2", file=sys.stderr)
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble')
      _p("testCreate 3", file=sys.stderr)
      resources.initialize_resources(resources.shared_resources()).run()
      _p("testCreate 4", file=sys.stderr)
      stamp_token = ensemble.get_stamp_token()
      _p("testCreate 5", file=sys.stderr)
      self.assertEqual(0, self.evaluate(stamp_token))
      _p("testCreate 6", file=sys.stderr)
      (_, num_trees, num_finalized_trees, num_attempted_layers,
       nodes_range) = ensemble.get_states()
      _p("testCreate 7", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_trees))
      _p("testCreate 8", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_finalized_trees))
      _p("testCreate 9", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_attempted_layers))
      _p("testCreate 10", file=sys.stderr)
      self.assertAllEqual([0, 1], self.evaluate(nodes_range))
      _p("testCreate 11", file=sys.stderr)

  @test_util.run_deprecated_v1
  def testCreateWithProto(self):
    _p("testCreateWithProto 1", file=sys.stderr)
    with self.cached_session():
      _p("testCreateWithProto 2", file=sys.stderr)
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      _p("testCreateWithProto 3", file=sys.stderr)
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            bucketized_split {
              threshold: 21
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.4
              original_leaf {
                scalar: 7.14
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 7
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.7
              original_leaf {
                scalar: -4.375
              }
            }
          }
          nodes {
            leaf {
              scalar: 6.54
            }
          }
          nodes {
            leaf {
              scalar: 7.305
            }
          }
          nodes {
            leaf {
              scalar: -4.525
            }
          }
          nodes {
            leaf {
              scalar: -4.145
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 75
              threshold: 21
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -1.4
            }
          }
          nodes {
            leaf {
              scalar: -0.6
            }
          }
          nodes {
            leaf {
              scalar: 0.165
            }
          }
        }
        tree_weights: 0.15
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 6
          last_layer_node_start: 16
          last_layer_node_end: 19
        }
      """, ensemble_proto)
      _p("testCreateWithProto 4", file=sys.stderr)
      ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble',
          stamp_token=7,
          serialized_proto=ensemble_proto.SerializeToString())
      _p("testCreateWithProto 5", file=sys.stderr)
      resources.initialize_resources(resources.shared_resources()).run()
      _p("testCreateWithProto 6", file=sys.stderr)
      (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
       nodes_range) = ensemble.get_states()
      _p("testCreateWithProto 7", file=sys.stderr)
      self.assertEqual(7, self.evaluate(stamp_token))
      _p("testCreateWithProto 8", file=sys.stderr)
      self.assertEqual(2, self.evaluate(num_trees))
      _p("testCreateWithProto 9", file=sys.stderr)
      self.assertEqual(1, self.evaluate(num_finalized_trees))
      _p("testCreateWithProto 10", file=sys.stderr)
      self.assertEqual(6, self.evaluate(num_attempted_layers))
      _p("testCreateWithProto 11", file=sys.stderr)
      self.assertAllEqual([16, 19], self.evaluate(nodes_range))
      _p("testCreateWithProto 12", file=sys.stderr)

  @test_util.run_deprecated_v1
  def testSerializeDeserialize(self):
    _p("testSerializeDeserialize 1", file=sys.stderr)
    with self.cached_session():
      _p("testSerializeDeserialize 2", file=sys.stderr)
      # Initialize.
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble', stamp_token=5)
      _p("testSerializeDeserialize 3", file=sys.stderr)
      resources.initialize_resources(resources.shared_resources()).run()
      _p("testSerializeDeserialize 4", file=sys.stderr)
      (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
       nodes_range) = ensemble.get_states()
      _p("testSerializeDeserialize 5", file=sys.stderr)
      self.assertEqual(5, self.evaluate(stamp_token))
      _p("testSerializeDeserialize 6", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_trees))
      _p("testSerializeDeserialize 7", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_finalized_trees))
      _p("testSerializeDeserialize 8", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_attempted_layers))
      _p("testSerializeDeserialize 9", file=sys.stderr)
      self.assertAllEqual([0, 1], self.evaluate(nodes_range))
      _p("testSerializeDeserialize 10", file=sys.stderr)

      # Deserialize.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      _p("testSerializeDeserialize 11", file=sys.stderr)
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 75
              threshold: 21
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -1.4
            }
          }
          nodes {
            leaf {
              scalar: -0.6
            }
          }
          nodes {
            leaf {
              scalar: 0.165
            }
          }
        }
        tree_weights: 0.5
        tree_metadata {
          num_layers_grown: 4  # it's fake intentionally.
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 5
          last_layer_node_start: 3
          last_layer_node_end: 7
        }
      """, ensemble_proto)
      _p("testSerializeDeserialize 12", file=sys.stderr)
      with ops.control_dependencies([
          ensemble.deserialize(
              stamp_token=3,
              serialized_proto=ensemble_proto.SerializeToString())
      ]):
        _p("testSerializeDeserialize 13", file=sys.stderr)
        (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
         nodes_range) = ensemble.get_states()
        _p("testSerializeDeserialize 14", file=sys.stderr)
      _p("testSerializeDeserialize 15", file=sys.stderr)
      self.assertEqual(3, self.evaluate(stamp_token))
      _p("testSerializeDeserialize 16", file=sys.stderr)
      self.assertEqual(1, self.evaluate(num_trees))
      _p("testSerializeDeserialize 17", file=sys.stderr)
      # This reads from metadata, not really counting the layers.
      self.assertEqual(5, self.evaluate(num_attempted_layers))
      _p("testSerializeDeserialize 18", file=sys.stderr)
      self.assertEqual(0, self.evaluate(num_finalized_trees))
      _p("testSerializeDeserialize 19", file=sys.stderr)
      self.assertAllEqual([3, 7], self.evaluate(nodes_range))
      _p("testSerializeDeserialize 20", file=sys.stderr)


      # Serialize.
      new_ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      _p("testSerializeDeserialize 21", file=sys.stderr)
      new_stamp_token, new_serialized = ensemble.serialize()
      _p("testSerializeDeserialize 22", file=sys.stderr)
      self.assertEqual(3, self.evaluate(new_stamp_token))
      _p("testSerializeDeserialize 23", file=sys.stderr)
      new_ensemble_proto.ParseFromString(new_serialized.eval())
      _p("testSerializeDeserialize 24", file=sys.stderr)
      self.assertProtoEquals(ensemble_proto, new_ensemble_proto)
      _p("testSerializeDeserialize 25", file=sys.stderr)


if __name__ == '__main__':
  googletest.main()
