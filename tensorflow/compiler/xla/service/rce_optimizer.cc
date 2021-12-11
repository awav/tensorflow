// License TODO ....

#include "tensorflow/compiler/xla/service/rce_optimizer.h"

#include <stdlib.h>
#include <istream>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class RceOptimizerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit RceOptimizerVisitor() {}

  Status HandleReduce(HloInstruction* dot) override;
  Status HandleGetTupleElement(HloInstruction *get_tuple_element) override;
};

}  // namespace

Status RceOptimizerVisitor::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  CHECK(Match(get_tuple_element, m::GetTupleElement()));
  HloInstruction *op;
  if (Match(get_tuple_element, m::GetTupleElement(m::Tuple(m::Op(&op))))) {
    auto tuple = get_tuple_element->operand(0);
    if (tuple->operand_count() == 1 && tuple->user_count() == 1) {
      return ReplaceInstruction(get_tuple_element, op);
    }
  }
}

Status RceOptimizerVisitor::HandleReduce(HloInstruction* reduce) {
  CHECK(Match(reduce, m::Reduce()));
  HloInstruction* op; 
  if (Match(reduce, m::Reduce(m::Reshape(m::Op(&op)), m::Constant()))) {
    auto reshape = reduce->operand(0);
    if (ShapeUtil::Equal(reduce->shape(), op->shape())) {
      return ReplaceInstruction(reduce, op);
    }
  }
}

StatusOr<bool> RceOptimizer::Run(HloModule* module) {
  RceOptimizerVisitor visitor;
  LOG(INFO) << "Running RCE optimizer ...";
  return visitor.RunOnModule(module);
}

}  // namespace xla
