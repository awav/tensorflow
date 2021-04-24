// License TODO ....
#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

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

  bool OperandShouldBeSplit(HloInstruction* inst);

  Status HandleDot(HloInstruction* dot) override;
};

}  // namespace

bool IntermediateTensorSplitterVisitor::OperandShouldBeSplit(
    HloInstruction* inst) {
  return ShapeUtil::ElementsIn(inst->shape()) > max_intermediate_size;
}

Status IntermediateTensorSplitterVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // Check if this dot

  // TODO
  LOG(INFO) << "Hello world!";
}

StatusOr<bool> IntermediateTensorSplitter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a good default
  IntermediateTensorSplitterVisitor visitor(1000 * 1000);
  return visitor.RunOnModule(module);
}

}  // namespace xla
