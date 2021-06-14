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

  bool DimsAreIndexes(HloInstruction* inst);

  bool IsTrivialBroadcast(HloInstruction* broadcast) {
    return DimsAreIndexes(broadcast);
  }

  bool IsTrivialTranspose(HloInstruction* transpose) {
    return DimsAreIndexes(transpose); /* transpose is subset of broadcast ... */
  }

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleConvert(HloInstruction* convert) override;
};

}  // namespace

bool BroadcastSimplifierVisitor::DimsAreIndexes(HloInstruction* inst) {
  for (int64 i = 0; i < inst->dimensions().size(); i++) {
    if (i != inst->dimensions(i)) {
      return false;
    }
  }
  return true;
}

Status BroadcastSimplifierVisitor::HandleBroadcast(HloInstruction* broadcast) {
  HloInstruction* op;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&op))));

  if (ShapeUtil::Equal(broadcast->shape(), op->shape()) &&
      IsTrivialBroadcast(broadcast)) {
    // This broadcast does nothing, remove it
    return ReplaceInstruction(broadcast, op);
  }
}

Status BroadcastSimplifierVisitor::HandleTranspose(HloInstruction* transpose) {
  HloInstruction* op;
  CHECK(Match(transpose, m::Transpose(m::Op(&op))));

  // We need to check the shape, since the transpose might modify the physical
  // layout, in which case we might loose information.
  if (ShapeUtil::Equal(transpose->shape(), op->shape()) &&
      IsTrivialTranspose(transpose)) {
    // This transpose does nothing, remove it
    return ReplaceInstruction(transpose, op);
  }
}

Status BroadcastSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  HloInstruction* op;
  CHECK(Match(reshape, m::Reshape(m::Op(&op))));

  // TODO: Does the physical layout matter for reshapes? I don't
  // think it does, but this might be something to investigate in
  // the future if problems arise.
  if (ShapeUtil::Equal(reshape->shape(), op->shape())) {
    // This reshape does nothing, remove it
    return ReplaceInstruction(reshape, op);
  }
}

Status BroadcastSimplifierVisitor::HandleConvert(HloInstruction* convert) {
  HloInstruction* op;
  CHECK(Match(convert, m::Convert(m::Op(&op))));

  if (ShapeUtil::Equal(convert->shape(), op->shape())) {
    // This convert does nothing, remove it
    return ReplaceInstruction(convert, op);
  }
}

StatusOr<bool> BroadcastSimplifier::Run(HloModule* module) {
  BroadcastSimplifierVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
