// License TODO ....

#include "tensorflow/compiler/xla/service/rewriting_optimizer.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = match;

class RewritingOptimizerTest : public HloTestBase {};

// Test (AB)C => A(BC)
TEST_F(RewritingOptimizerTest, MatrixVectorDot) {
  EXPECT_EQ(true, false);
}

}
} // namespace xla
