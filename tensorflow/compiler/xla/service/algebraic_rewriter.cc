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
                           int64* x_reduce_dim, int64* y_dim,
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

  // Check add or sub
  HloInstruction *lhs, *rhs;
  if (Match(add_or_sub, m::Add(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = false;
  else if (Match(add_or_sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))))
    *is_sub = true;
  else
    return false;

  // Check broadcast
  if (!Match(lhs, m::Broadcast(m::Op(x))) ||
      !Match(rhs, m::Broadcast(m::Op(y))))
    return false;

  // Check the constants are correct
  if (ShapeUtil::ElementsIn(reduce_init->shape()) != 1) return false;
  if (!reduce_init->literal().IsZero({0})) return false;

  if (ShapeUtil::ElementsIn(power_const->shape()) != 1) return false;
  if (!power_const->literal().Get<float>({0}) == 2.0) return false;

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
  if (lhs->shape().rank() != 3) return false;
  int64 rank = lhs->shape().rank();
  *x_dim = -1;
  *y_dim = -1;
  for (int64 i = 0; i < rank; i++) {
    for (int64 j = 0; j < rank; j++) {
      if (i == j) continue;
      if (i == reduce_dim || j == reduce_dim) continue;
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

  HloInstruction *x, *y;
  bool is_sub;
  int64 x_dim, x_reduce_dim, x_dot_dim, y_dim, y_reduce_dim, y_dot_dim;
  if (!MatchDistanceMatrix(reduce, &x, &y, &is_sub, &x_dim, &x_reduce_dim,
                           &y_dim, &y_reduce_dim))
    return Status::OK();

  // constants
  HloInstruction* zero = reduce->parent()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  HloInstruction* two = reduce->parent()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2)));

  // squares
  HloInstruction* two_x = reduce->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(x->shape(), two, {}));
  HloInstruction* x_squared = reduce->parent()->AddInstruction(
      HloInstruction::CreateBinary(x->shape(), HloOpcode::kPower, x, two_x));
  HloInstruction* two_y = reduce->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(y->shape(), two, {}));
  HloInstruction* y_squared = reduce->parent()->AddInstruction(
      HloInstruction::CreateBinary(y->shape(), HloOpcode::kPower, y, two_y));

  // reduce the squares
  HloComputation* reduce_sum = reduce->called_computations()[0];

  Shape x_reduce_shape = ShapeUtil::MakeShape(x_squared->shape().element_type(),
                                              x_squared->shape().dimensions());
  x_reduce_shape.DeleteDimension(x_reduce_dim);
  HloInstruction* x_reduce =
      reduce->parent()->AddInstruction(HloInstruction::CreateReduce(
          x_reduce_shape, x_squared, zero, {x_reduce_dim}, reduce_sum));

  Shape y_reduce_shape = ShapeUtil::MakeShape(y_squared->shape().element_type(),
                                              y_squared->shape().dimensions());
  y_reduce_shape.DeleteDimension(y_reduce_dim);
  HloInstruction* y_reduce =
      reduce->parent()->AddInstruction(HloInstruction::CreateReduce(
          y_reduce_shape, y_squared, zero, {y_reduce_dim}, reduce_sum));

  // x y outer product
  HloInstruction* xy;
  if (x_dim == 0 && y_dim == 1) {
    Shape xy_shape = ShapeUtil::MakeShape(x->shape().element_type(), {});
    for (int64 i = 0; i < x->shape().rank(); i++)
      if (i != x_reduce_dim) xy_shape.add_dimensions(x->shape().dimensions(i));
    for (int64 i = 0; i < y->shape().rank(); i++)
      if (i != y_reduce_dim) xy_shape.add_dimensions(y->shape().dimensions(i));

    PrecisionConfig conf;
    DotDimensionNumbers dnums;
    dnums.add_lhs_contracting_dimensions(x_reduce_dim);
    dnums.add_rhs_contracting_dimensions(y_reduce_dim);
    xy = reduce->parent()->AddInstruction(
        HloInstruction::CreateDot(xy_shape, x, y, dnums, conf));
  } else if (x_dim == 1 && y_dim == 0) {
    Shape xy_shape = ShapeUtil::MakeShape(x->shape().element_type(), {});
    for (int64 i = 0; i < y->shape().rank(); i++)
      if (i != x_reduce_dim) xy_shape.add_dimensions(y->shape().dimensions(i));
    for (int64 i = 0; i < x->shape().rank(); i++)
      if (i != y_reduce_dim) xy_shape.add_dimensions(x->shape().dimensions(i));

    PrecisionConfig conf;
    DotDimensionNumbers dnums;
    dnums.add_lhs_contracting_dimensions(y_reduce_dim);
    dnums.add_rhs_contracting_dimensions(x_reduce_dim);
    xy = reduce->parent()->AddInstruction(
        HloInstruction::CreateDot(xy_shape, y, x, dnums, conf));
  } else {
    // failed pre-condition :/
    CHECK((x_dim == 0 && y_dim == 1) || (x_dim == 1 && y_dim == 0));
  }

  HloInstruction* x_broadcast = reduce->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(xy->shape(), x_reduce, {x_dim}));
  HloInstruction* y_broadcast = reduce->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(xy->shape(), y_reduce, {y_dim}));

  HloInstruction* x_y_sum =
      reduce->parent()->AddInstruction(HloInstruction::CreateBinary(
          xy->shape(), HloOpcode::kAdd, x_broadcast, y_broadcast));

  ReplaceInstruction(
      reduce, reduce->parent()->AddInstruction(HloInstruction::CreateBinary(
                  xy->shape(), HloOpcode::kAdd, x_y_sum, xy)));
}

StatusOr<bool> AlgebraicRewriter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  // int64 split_size = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  AlgebraicRewriterVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
