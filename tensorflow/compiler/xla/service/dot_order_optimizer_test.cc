// License TODO ....

#include "tensorflow/compiler/xla/service/dot_order_optimizer.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = match;

class DotOrderOptimizerTest : public HloTestBase {};

// Test (AB)C => A(BC)
TEST_F(DotOrderOptimizerTest, MatrixVectorDotLhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape mat_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, mat_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, mat_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, mat_shape, "c"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {1000, 1000});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  DotDimensionNumbers dnums_abc;
  dnums_abc.add_lhs_contracting_dimensions(1);
  dnums_abc.add_rhs_contracting_dimensions(0);
  Shape abc_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  builder.AddInstruction(HloInstruction::CreateDot(abc_shape, ab, c, dnums_abc,
                                                   DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(
      Match(computation->root_instruction(),
            m::Dot(m::Dot(m::Op().Is(a), m::Op().Is(b)), m::Op().Is(c))));

  DotOrderOptimizer optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(
      Match(computation->root_instruction(),
            m::Dot(m::Op().Is(a), m::Dot(m::Op().Is(b), m::Op().Is(c)))));
}

// TEST A(BC) => (AB)C
TEST_F(DotOrderOptimizerTest, MatrixVectorDotRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape mat_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, mat_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, mat_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, mat_shape, "c"));

  DotDimensionNumbers dnums_bc;
  dnums_bc.add_lhs_contracting_dimensions(1);
  dnums_bc.add_rhs_contracting_dimensions(1);
  Shape bc_shape = ShapeUtil::MakeShape(F32, {1000, 1000});
  HloInstruction* bc = builder.AddInstruction(HloInstruction::CreateDot(bc_shape, b, c, dnums_bc,
                                                   DefaultPrecisionConfig(2)));

  DotDimensionNumbers dnums_abc;
  dnums_abc.add_lhs_contracting_dimensions(0);
  dnums_abc.add_rhs_contracting_dimensions(0);
  Shape abc_shape = ShapeUtil::MakeShape(F32, {2, 1000});
  builder.AddInstruction(HloInstruction::CreateDot(abc_shape, a, bc, dnums_abc,
                                                   DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(
      Match(computation->root_instruction(),
            m::Dot(m::Op().Is(a), m::Dot(m::Op().Is(b), m::Op().Is(c)))));

  DotOrderOptimizer optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(
      Match(computation->root_instruction(),
            m::Dot(m::Dot(m::Op().Is(a), m::Op().Is(b)), m::Op().Is(c))));
}

}  // namespace
}  // namespace xla
