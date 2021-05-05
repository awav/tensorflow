// License TODO ....
#include "tensorflow/compiler/xla/service/broadcast_simplifier.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class BroadcastSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  explicit BroadcastSimplifierVisitor() {}

  Status HandleBroadcast(HloInstruction* broadcast) override;
};

}  // namespace

Status BroadcastSimplifierVisitor::HandleBroadcast(HloInstruction* broadcast) {
  HloInstruction *op;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&op))));

  if (ShapeUtil::Equal(broadcast->shape(), op->shape())) {
    // This broadcast is useless, remove it
    return ReplaceInstruction(broadcast, op);
  }
}

StatusOr<bool> BroadcastSimplifier::Run(HloModule* module) {
  BroadcastSimplifierVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
