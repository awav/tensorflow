// License TODO ....
#include "tensorflow/compiler/xla/service/algebraic_rewriter.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class AlgebraicRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AlgebraicRewriterVisitor() {}

  bool MatchDistanceMatrix(HloInstruction* reduce, HloInstruction** x,
                           HloInstruction** y, bool* is_sub);

  Status HandleReduce(HloInstruction* reduce) override;
};

}  // namespace

bool AlgebraicRewriterVisitor::MatchDistanceMatrix(HloInstruction* reduce,
                                                   HloInstruction** x,
                                                   HloInstruction** y,
                                                   bool* is_sub) {
  HloInstruction* reduce_op;
  HloInstruction* reduce_init;
  if (!Match(reduce, m::Reduce(m::Op(&reduce_op), m::Op(&reduce_init))))
    return false;
  HloInstruction* reduce_init_const;
  if (!Match(reduce_init, m::Constant(&reduce_init_const)) &&
      !Match(reduce_init, m::Convert(m::Constant(&reduce_init_const))))
    return false;

  if (ShapeUtil::ElementsIn(reduce_init_const->shape()) != 1) return false;
  if (!reduce_init_const->literal().IsZero({0})) return false;

  // TODO: Verify the reduce computation is indeed a sum ...
  HloComputation* reduce_computation = reduce->called_computations()[0];

  // TODO: Redo this without the converts, since we now have a
  // pass to remove them ...
  // HloInstruction* sub_or_add;
  // if (!Match(reduce_op, m::Add(&sub_or_add)) &&
  //     !Match(reduce_op, m::Convert(m::Add(&sub_or_add))) &&
  //     !Match(reduce_op, m::Subtract(&sub_or_add)) &&
  //     !Match(reduce_op, m::Convert(m::Subtract(&sub_or_add))))
  //   return false;

  // HloInstruction* lhs;
  // HloInstruction* rhs;
  // if (Match(sub_or_add, m::Add(&sub_or_add, m::Op(&lhs), m::Op(&rhs)))) {
  //   *is_sub = false;
  // } else if (Match(sub_or_add, m::Subtract(&sub_or_add, m::Op(&lhs), m::Op(&rhs)))) {
  //   *is_sub = true;
  // } else {
  //   return false;
  // }

  if (!Match(lhs, m::Broadcast(m::Op(x))) ||
      !Match(rhs, m::Broadcast(m::Op(y))))
    return false;

  // TODO: Check that the broad-casts are of the appropriate dimensions

  return true;
}

Status AlgebraicRewriteVisitor::HandleReduce(HloInstruction* reduce) {
  LOG(INFO) << "Hello world!";

  // match (x +/- y)
  HloInstruction* x;
  HloInstruction* y;
}

StatusOr<bool> AlgebraicRewriter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  // int64 split_size = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  AlgebraicRewriterVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
