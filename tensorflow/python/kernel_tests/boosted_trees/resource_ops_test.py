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

from google.protobuf import text_format

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


class ResourceOpsTest(test_util.TensorFlowTestCase):
  """Tests resource_ops."""

  @test_util.run_deprecated_v1
  def testCreate(self):
    _p("testCreate 1", file=sys.stderr)
    with self.cached_session():
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble')
      resources.initialize_resources(resources.shared_resources()).run()
      stamp_token = ensemble.get_stamp_token()
      self.assertEqual(0, self.evaluate(stamp_token))
      (_, num_trees, num_finalized_trees, num_attempted_layers,
       nodes_range) = ensemble.get_states()
      self.assertEqual(0, self.evaluate(num_trees))
      self.assertEqual(0, self.evaluate(num_finalized_trees))
      self.assertEqual(0, self.evaluate(num_attempted_layers))
      self.assertAllEqual([0, 1], self.evaluate(nodes_range))
    _p("testCreate 2", file=sys.stderr)

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


if __name__ == '__main__':
  googletest.main()
