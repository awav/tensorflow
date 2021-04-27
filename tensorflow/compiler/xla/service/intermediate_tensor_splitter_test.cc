// License TODO ....

#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = match;

class IntermediateTensorSplitterTest : public HloTestBase {
 protected:
  const int64 MAX_SIZE = 1000 * 1000;  // TODO: This might change ...

  const int64 max_op_size_in_graph(HloInstruction* inst) {
    int64 max_size = 0;
    max_size = std::max(max_size, ShapeUtil::ElementsIn(inst->shape()));
    for (HloInstruction* op : inst->operands()) {
      max_size = std::max(max_size, max_op_size_in_graph(op));
    }
    return max_size;
  }
};

// Test the most basic case: exp(AB^T)v
TEST_F(IntermediateTensorSplitterTest, BasicCaseLhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {2000, 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {1000});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {2000, 1000});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(1);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {2000});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, v, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))), m::Op().Is(v))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) > MAX_SIZE);

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              MAX_SIZE);
}

// Test the most basic rhs case: exp(AB^T)v
TEST_F(IntermediateTensorSplitterTest, BasicCaseRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {2000, 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {1000});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {2000, 1000});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(1);
  Shape final_shape = ShapeUtil::MakeShape(F32, {2000});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, v, exp_ab, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Op().Is(v), m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) > MAX_SIZE);

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              MAX_SIZE);
}

}  // namespace
}  // namespace xla
