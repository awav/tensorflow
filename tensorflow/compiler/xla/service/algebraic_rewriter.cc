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

  Status HandlePower(HloInstruction* pow) override;
};

}  // namespace

bool AlgebraicRewriterVisitor::MatchDistanceMatrix(HloInstruction* power,
                                                   HloInstruction** x,
                                                   HloInstruction** y,
                                                   bool* is_sub) {
  // Check up to reduce
  HloInstruction* _;
  HloInstruction* reduce;
  HloInstruction* add_or_sub;
  HloInstruction* reduce_init;
  HloInstruction* power_const;
  if (!Match(power, m::Power(m::Op(&reduce), m::Broadcast(m::Constant(&power_const)))))
    return false;
  LOG(INFO) << "matched power";
  if (!Match(reduce, m::Reduce(m::Op(&add_or_sub), m::Constant(&reduce_init))))
    return false;
  LOG(INFO) << "matched reduce";

  // Check add or sub
  HloInstruction *lhs, *rhs;
  if (Match(add_or_sub, m::Add(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = false;
  else if (Match(add_or_sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = true;
  else
    return false;
  LOG(INFO) << "matched addition/ subtraction";

  // Check broadcast
  if (!Match(lhs, m::Broadcast(m::Op(x))) ||
      !Match(rhs, m::Broadcast(m::Op(y))))
    return false;
  LOG(INFO) << "matched broadcast";

  // Check the constants are correct
  if (ShapeUtil::ElementsIn(reduce_init->shape()) != 1) return false;
  if (!reduce_init->literal().IsZero({0})) return false;
  LOG(INFO) << "matched reduce init = 0";

  if (ShapeUtil::ElementsIn(power_const->shape()) != 1) return false;
  if (!power_const->literal().Get<float>({0}) == 2.0) return false;
  LOG(INFO) << "matched power = 2";

  // TODO: Check the broadcast + reduce dimensions are correct

  // TODO: Check reduce computation is add

  return true;
}

Status AlgebraicRewriterVisitor::HandlePower(HloInstruction* pow) {
  LOG(INFO) << "Hello world!";

  HloInstruction *x, *y;
  bool is_sub;
  if (!MatchDistanceMatrix(pow, &x, &y, &is_sub)) return Status::OK();

  LOG(INFO) << "Matched dist matrix! Is sub = " << (is_sub ? "yes" : "no");
}

StatusOr<bool> AlgebraicRewriter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  // int64 split_size = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  AlgebraicRewriterVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
