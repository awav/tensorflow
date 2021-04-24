// License TODO ....
#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class IntermediateTensorSplitterVisitor : public DfsHloRewriteVisitor {
  int max_intermediate_size;

 public:
  explicit IntermediateTensorSplitterVisitor(int max_intermediate_size)
      : max_intermediate_size(max_intermediate_size) {}

  // Determine if an operand is large enough such that we are
  // interested in splitting it.
  bool OperandShouldBeSplit(HloInstruction* inst);

  // Determine if an operand can be split by traversing it's
  // inputs until a splittable node is found.
  bool OperandCanBeSplit(HloInstruction* inst);

  // Matches any pointwise unary operator which has no side effects.
  bool MatchPointwiseUnary(HloInstruction* inst,
                           HloInstruction** operand = nullptr);

  // Determine the best dimesion to split on, excluding a given one.
  int64 BestSplitDim(HloInstruction* inst, absl::Span<const int64> excluded);

  // Collect computation for the instruction we want to split
  // and split the parameters. The parameters are returned pre-
  // split such that they can be used verbatim inside a call.
  // The returned instruction is the root instruction of the
  // computation.
  StatusOr<HloInstruction*> BuildComputationAndParameters(
      HloInstruction* inst, int64 split_dim, int64 split_size,
      HloComputation::Builder* builder,
      std::vector<std::vector<HloInstruction*>>* parameters);

  Status HandleDot(HloInstruction* dot) override;
};

}  // namespace

bool IntermediateTensorSplitterVisitor::OperandShouldBeSplit(
    HloInstruction* inst) {
  return ShapeUtil::ElementsIn(inst->shape()) > max_intermediate_size;
}

bool IntermediateTensorSplitterVisitor::MatchPointwiseUnary(
    HloInstruction* inst, HloInstruction** operand) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() == 1) {
    if (operand != NULL) {
      *operand = inst->mutable_operand(0);
    }
    return true;
  } else {
    return false;
  }
}

bool IntermediateTensorSplitterVisitor::OperandCanBeSplit(
    HloInstruction* inst) {
  HloInstruction* next;

  if (Match(inst, m::Dot(m::Op(), m::Op()))) {
    // Base case: A Dot produces this large intermediate tensor
    // TODO: Support more cases (most importantly broadcast..)
    return true;
  } else if (MatchPointwiseUnary(inst, &next)) {
    return OperandCanBeSplit(next);
  } else if (Match(inst, m::Transpose(m::Op(&next)))) {
    return OperandCanBeSplit(next);
  } else {
    return false;
  }
}

int64 IntermediateTensorSplitterVisitor::BestSplitDim(
    HloInstruction* inst, absl::Span<const int64> excluded) {
  const Shape& shape = inst->shape();
  int64 best_dim = -1, best_size = 0;
  for (int64 i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) continue;
    if (shape.dimensions(i) > best_size) {
      best_size = shape.dimensions(i);
      best_dim = i;
    }
  }
  return best_dim;
}

StatusOr<HloInstruction*>
IntermediateTensorSplitterVisitor::BuildComputationAndParameters(
    HloInstruction* inst, int64 split_dim, int64 split_size,
    HloComputation::Builder* builder,
    std::vector<std::vector<HloInstruction*>>* parameters) {
  HloInstruction *operand, *lhs, *rhs;

  if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
    // For the dot we identify the parameter to split and then
    // Generate the final dot operation, as well as the operand
    // vector.
    CHECK(false);  // TODO
  } else if (MatchPointwiseUnary(inst, &operand)) {
    // For a unary operation recursively obtain a new operand and
    // clone the operation.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        BuildComputationAndParameters(operand, split_dim, split_size, builder,
                                      parameters));
    return builder->AddInstruction(inst->CloneWithNewOperands(
        new_operand->shape(), absl::MakeSpan(&new_operand, 1)));
  } else if (Match(inst, m::Transpose(m::Op(&operand)))) {
    // For a transpose, the transpose might change which dimension is
    // being split. So we obtain the new split dimension and then
    // recursively a new operand to make a clone.
    int64 operand_split_dim = inst->dimensions(split_dim);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        BuildComputationAndParameters(operand, operand_split_dim, split_size,
                                      builder, parameters));
    return builder->AddInstruction(inst->CloneWithNewOperands(
        new_operand->shape(), absl::MakeSpan(&new_operand, 1)));
  } else {
    // Invariant violation
    // TODO: Is there a more idiomatic way to return a bad status?
    CHECK(false);
  }
}

Status IntermediateTensorSplitterVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  // Check if this dot is large enough to be split
  if (OperandShouldBeSplit(lhs) && OperandCanBeSplit(lhs)) {
    LOG(INFO) << "Will attempt to split lhs";
    int64 split_dim =
        BestSplitDim(lhs, absl::MakeSpan(dnums.lhs_contracting_dimensions()));

    HloComputation::Builder builder("intermediate_tensor_computation_builde");
    std::vector<std::vector<HloInstruction*>> parameters;
    // TODO: Make the split size configurable (and smarter ...)
    TF_ASSIGN_OR_RETURN(HloInstruction * comp_root,
                        BuildComputationAndParameters(lhs, split_dim, 1000,
                                                      &builder, &parameters));

    // TODO !
  } else if (OperandShouldBeSplit(rhs) && OperandCanBeSplit(rhs)) {
    LOG(INFO) << "Will attempt to split rhs: TODO";
  }
}

StatusOr<bool> IntermediateTensorSplitter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a good default
  IntermediateTensorSplitterVisitor visitor(1000 * 1000);
  return visitor.RunOnModule(module);
}

}  // namespace xla
