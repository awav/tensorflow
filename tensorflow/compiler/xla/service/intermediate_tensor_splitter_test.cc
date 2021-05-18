// License TODO ....

#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

#include "tensorflow/compiler/xla/debug_options_flags.h"
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
  const int64 max_size() {
    return GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  }

  const int64 large_dim() { return 2 * max_size() / 10000; }

  const int64 max_op_size_in_graph(HloInstruction* inst) {
    int64 max_size = 0;
    max_size = std::max(max_size, ShapeUtil::ElementsIn(inst->shape()));
    for (HloInstruction* op : inst->operands()) {
      max_size = std::max(max_size, max_op_size_in_graph(op));
    }
    return max_size;
  }

  string replace_all_in_string(string original, string find, string replace) {
    int len = find.length();
    size_t index = 0;
    while (true) {
      index = original.find(find, index);
      if (index == std::string::npos) break;
      original.replace(index, len, replace);
      index += len;
    }
    return original;
  }
};

// Test the most basic case: exp(AB^T)v
TEST_F(IntermediateTensorSplitterTest, BasicCaseLhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(1);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, v, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))), m::Op().Is(v))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test the most basic rhs case: exp(AB^T)v
TEST_F(IntermediateTensorSplitterTest, BasicCaseRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(1);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, v, exp_ab, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Op().Is(v), m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test the case where the to split dimension lies on the
// rhs of the source dot
TEST_F(IntermediateTensorSplitterTest, BasicSplitDotOnRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, v, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))), m::Op().Is(v))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test broadcast instructions as source
TEST_F(IntermediateTensorSplitterTest, Broadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape param_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  std::vector<int64> dims = {};
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, p, absl::MakeSpan(dims)));

  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(1, v_shape, "v"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      v_shape, broadcast, v, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Broadcast(m::Op().Is(p)), m::Op().Is(v))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test broadcast instructions as source when split dim
// is a real dimension
TEST_F(IntermediateTensorSplitterTest, BroadcastSplitOnOperandDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  std::vector<int64> dims = {0};
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, p, absl::MakeSpan(dims)));

  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(1, v_shape, "v"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(0);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      v_shape, broadcast, v, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Broadcast(m::Op().Is(p)), m::Op().Is(v))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test iota with iota dimension along split
TEST_F(IntermediateTensorSplitterTest, IotaSplitAlongIotaDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape iota_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});

  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(iota_shape, 0));
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      param_shape, iota, param, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Iota().Is(iota), m::Op().Is(param))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test iota with non-iota dimension along split
TEST_F(IntermediateTensorSplitterTest, IotaSplitAlongNonIotaDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape iota_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});

  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(iota_shape, 1));
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      param_shape, iota, param, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Iota().Is(iota), m::Op().Is(param))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test single argument reduce (e.g. max)
TEST_F(IntermediateTensorSplitterTest, SingleOperandReduce) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloComputation::Builder max_builder(TestName() + ".max");

  Shape empty_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* x = max_builder.AddInstruction(
      HloInstruction::CreateParameter(0, empty_shape, "x"));
  HloInstruction* y = max_builder.AddInstruction(
      HloInstruction::CreateParameter(1, empty_shape, "y"));
  max_builder.AddInstruction(
      HloInstruction::CreateBinary(empty_shape, HloOpcode::kMaximum, x, y));
  HloComputation* max = m->AddEmbeddedComputation(max_builder.Build());

  Shape big_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateIota(big_shape, 0));

  HloInstruction* init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  Shape small_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(
      HloInstruction::CreateReduce(small_shape, a, init, {1}, max));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Reduce(m::Op().Is(a), m::Op().Is(init))));
  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) >
              max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(computation->root_instruction()) <=
              max_size());
}

// Test multi argument reduce (e.g. argmax)
TEST_F(IntermediateTensorSplitterTest, MultiOperandReduce) {
  const string module_str = R"(
HloModule a_inference_arg_max_test_29__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.35

%minmax_func.17 (lhs_value.18: f32[], lhs_index.19: s32[], rhs_value.20: f32[], rhs_index.21: s32[]) -> (f32[], s32[]) {
  %lhs_value.18 = f32[] parameter(0)
  %rhs_value.20 = f32[] parameter(2)
  %compare.22 = pred[] compare(f32[] %lhs_value.18, f32[] %rhs_value.20), direction=GE
  %select.23 = f32[] select(pred[] %compare.22, f32[] %lhs_value.18, f32[] %rhs_value.20)
  %compare.25 = pred[] compare(f32[] %lhs_value.18, f32[] %rhs_value.20), direction=EQ
  %lhs_index.19 = s32[] parameter(1)
  %rhs_index.21 = s32[] parameter(3)
  %minimum.26 = s32[] minimum(s32[] %lhs_index.19, s32[] %rhs_index.21)
  %select.24 = s32[] select(pred[] %compare.22, s32[] %lhs_index.19, s32[] %rhs_index.21)
  %select.27 = s32[] select(pred[] %compare.25, s32[] %minimum.26, s32[] %select.24)
  ROOT %tuple.28 = (f32[], s32[]) tuple(f32[] %select.23, s32[] %select.27)
}

ENTRY %a_inference_arg_max_test_29__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.35 (arg0.1: f32[2000,1000], arg1.2: f32[2000,1000]) -> s64[2000,2000] {
  %arg0.1 = f32[2000,1000]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.3 = f32[2000,1000]{1,0} reshape(f32[2000,1000]{1,0} %arg0.1)
  %slice.5 = f32[2000,1000]{1,0} slice(f32[2000,1000]{1,0} %reshape.3), slice={[0:2000], [0:1000]}, metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=224}
  %reshape.6 = f32[1,2000,1000]{2,1,0} reshape(f32[2000,1000]{1,0} %slice.5), metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=224}
  %reshape.9 = f32[2000,1000]{1,0} reshape(f32[1,2000,1000]{2,1,0} %reshape.6), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %broadcast.10 = f32[2000,2000,1000]{2,1,0} broadcast(f32[2000,1000]{1,0} %reshape.9), dimensions={1,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %arg1.2 = f32[2000,1000]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.4 = f32[2000,1000]{1,0} reshape(f32[2000,1000]{1,0} %arg1.2)
  %slice.7 = f32[2000,1000]{1,0} slice(f32[2000,1000]{1,0} %reshape.4), slice={[0:2000], [0:1000]}, metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=224}
  %reshape.8 = f32[2000,1,1000]{2,1,0} reshape(f32[2000,1000]{1,0} %slice.7), metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=224}
  %reshape.11 = f32[2000,1000]{1,0} reshape(f32[2000,1,1000]{2,1,0} %reshape.8), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %broadcast.12 = f32[2000,2000,1000]{2,1,0} broadcast(f32[2000,1000]{1,0} %reshape.11), dimensions={0,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %subtract.13 = f32[2000,2000,1000]{2,1,0} subtract(f32[2000,2000,1000]{2,1,0} %broadcast.10, f32[2000,2000,1000]{2,1,0} %broadcast.12), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %iota.16 = s32[2000,2000,1000]{2,1,0} iota(), iota_dimension=2, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %constant.14 = f32[] constant(-inf), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %constant.15 = s32[] constant(0), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %reduce.29 = (f32[2000,2000]{1,0}, s32[2000,2000]{1,0}) reduce(f32[2000,2000,1000]{2,1,0} %subtract.13, s32[2000,2000,1000]{2,1,0} %iota.16, f32[] %constant.14, s32[] %constant.15), dimensions={2}, to_apply=%minmax_func.17, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %get-tuple-element.30 = s32[2000,2000]{1,0} get-tuple-element((f32[2000,2000]{1,0}, s32[2000,2000]{1,0}) %reduce.29), index=1, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %convert.31 = s64[2000,2000]{1,0} convert(s32[2000,2000]{1,0} %get-tuple-element.30), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %reshape.32 = s64[2000,2000]{1,0} reshape(s64[2000,2000]{1,0} %convert.31), metadata={op_name="XLA_Retvals"}
  %tuple.33 = (s64[2000,2000]{1,0}) tuple(s64[2000,2000]{1,0} %reshape.32), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.34 = s64[2000,2000]{1,0} get-tuple-element((s64[2000,2000]{1,0}) %tuple.33), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  string module_with_big_dims = replace_all_in_string(
      module_str, "2000", std::to_string(large_dim() / 3));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_with_big_dims));

  HloComputation* entry = module->entry_computation();

  EXPECT_TRUE(max_op_size_in_graph(entry->root_instruction()) > max_size());

  IntermediateTensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, module.get()));
  EXPECT_TRUE(result);

  EXPECT_TRUE(max_op_size_in_graph(entry->root_instruction()) <= max_size());
}

}  // namespace
}  // namespace xla
