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
                           HloInstruction** y, bool* is_sub, int64* x_dim,
                           int64* x_redzce_dim, int64* y_dim,
                           int64* y_reduce_dim);

  Status HandleReduce(HloInstruction* reduce) override;
};

}  // namespace

bool AlgebraicRewriterVisitor::MatchDistanceMatrix(
    HloInstruction* reduce, HloInstruction** x, HloInstruction** y,
    bool* is_sub, int64* x_dim, int64* x_reduce_dim, int64* y_dim,
    int64* y_reduce_dim) {
  // Check up to reduce
  HloInstruction* add_or_sub;
  HloInstruction* reduce_init;
  HloInstruction* power_const;
  if (!Match(reduce,
             m::Reduce(m::Power(m::Op(&add_or_sub),
                                m::Broadcast(m::Constant(&power_const))),
                       m::Constant(&reduce_init))))
    return false;
  LOG(INFO) << "match reduce + power";

  // Check add or sub
  HloInstruction *lhs, *rhs;
  if (Match(add_or_sub, m::Add(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = false;
  else if (Match(add_or_sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = true;
  else
    return false;
  LOG(INFO) << "match add or sub";

  // Check broadcast
  if (!Match(lhs, m::Broadcast(m::Op(x))) ||
      !Match(rhs, m::Broadcast(m::Op(y))))
    return false;
  LOG(INFO) << "match broadcasts";

  // Check the constants are correct
  if (ShapeUtil::ElementsIn(reduce_init->shape()) != 1) return false;
  if (!reduce_init->literal().IsZero({0})) return false;
  LOG(INFO) << "match reduce const";

  if (ShapeUtil::ElementsIn(power_const->shape()) != 1) return false;
  if (!power_const->literal().Get<float>({0}) == 2.0) return false;
  LOG(INFO) << "match power const";

  // Check the broadcast + reduce dimensions are correct
  int64 reduce_dim = reduce->dimensions(0);
  // the reduce dimension must NOT be a broadcasted one
  *x_reduce_dim = -1;
  *y_reduce_dim = -1;
  for (int64 i = 0; i < lhs->dimensions().size(); i++) {
    if (lhs->dimensions(i) == reduce_dim) *x_reduce_dim = i;
  }
  for (int64 i = 0; i < rhs->dimensions().size(); i++) {
    if (rhs->dimensions(i) == reduce_dim) *y_reduce_dim = i;
  }
  if (*x_reduce_dim == -1 || *y_reduce_dim == -1) return false;
  // there must be a pair of dims i =/= j such that:
  // x -> i : R, j : B and y -> i : B, j : R
  CHECK(ShapeUtil::Equal(lhs->shape(), rhs->shape()));
  if (lhs->shape().rank() < 3) return false;
  int64 rank = lhs->shape().rank();
  *x_dim = -1;
  *y_dim = -1;
  for (int64 i = 0; i < rank; i++) {
    for (int64 j = 0; j < rank; j++) {
      if (i == j) continue;
      if (absl::c_linear_search(lhs->dimensions(), i) &&
          !absl::c_linear_search(lhs->dimensions(), j) &&
          !absl::c_linear_search(rhs->dimensions(), i) &&
          absl::c_linear_search(rhs->dimensions(), j)) {
        *x_dim = i;
        *y_dim = j;
      }
    }
  }
  if (*x_dim == -1 || *y_dim == -1) return false;

  // TODO: Check reduce computation is add

  return true;
}

Status AlgebraicRewriterVisitor::HandleReduce(HloInstruction* reduce) {
  LOG(INFO) << "Hello world!";

  HloInstruction *x, *y;
  bool is_sub;
  int64 x_dim, x_reduce_dim, y_dim, y_reduce_dim;
  if (!MatchDistanceMatrix(reduce, &x, &y, &is_sub, &x_dim, &x_reduce_dim,
                           &y_dim, &y_reduce_dim))
    return Status::OK();

  LOG(INFO) << "Matched dist matrix! Is sub = " << (is_sub ? "yes" : "no");
}

StatusOr<bool> AlgebraicRewriter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  // int64 split_size = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  AlgebraicRewriterVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
