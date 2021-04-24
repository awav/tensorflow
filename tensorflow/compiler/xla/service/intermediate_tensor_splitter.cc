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
                           HloInstruction** operand = NULL);

  // Determine the best dimesion to split on, excluding a given one.
  int BestSplitDim(HloInstruction* inst, absl::Span<const int64> excluded);

  // Collect computation for the instruction we want to split
  // and split the parameters. The parameters are returned pre-
  // split such that they can be used verbatim inside a call.
  Status BuildComputationAndParameters(
      HloInstruction* inst, int split_dim, HloComputation::Builder& builder,
      std::vector<std::vector<HloInstruction*>>& parameters);

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

int IntermediateTensorSplitterVisitor::BestSplitDim(HloInstruction* inst, absl::Span<const int64> excluded) {
  const Shape& shape = inst->shape();
  int best_dim = -1, best_size = 0;
  for (int64 i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) continue;
    if (shape.dimensions(i) > best_size) {
      best_size = shape.dimensions(i);
      best_dim = i;
    }
  }
  return best_dim;
}

Status IntermediateTensorSplitterVisitor::BuildComputationAndParameters(
      HloInstruction* inst, int split_dim, HloComputation::Builder& builder,
      std::vector<std::vector<HloInstruction*>>& parameters) {
  // TODO

  return Status::OK();
}

Status IntermediateTensorSplitterVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  // Check if this dot is large enough to be split
  if (OperandShouldBeSplit(lhs) && OperandCanBeSplit(lhs)) {
    LOG(INFO) << "Will attempt to split lhs";
    int split_dim = BestSplitDim(lhs, absl::MakeSpan(dnums.lhs_contracting_dimensions()));

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
