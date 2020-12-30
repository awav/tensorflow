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
    _p("testCreate 2", file=sys.stderr)
    with self.cached_session():
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble')
      resources.initialize_resources(resources.shared_resources()).run()
    _p("testCreate 3", file=sys.stderr)


if __name__ == '__main__':
  googletest.main()
