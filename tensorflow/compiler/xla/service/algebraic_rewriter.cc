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
                           HloInstruction** y, bool* is_sub, int64_t* x_dim,
                           int64_t* x_reduce_dim, int64_t* y_dim,
                           int64_t* y_reduce_dim);

  Status HandleReduce(HloInstruction* reduce) override;
};

}  // namespace

bool AlgebraicRewriterVisitor::MatchDistanceMatrix(
    HloInstruction* reduce, HloInstruction** x, HloInstruction** y,
    bool* is_sub, int64_t* x_dim, int64_t* x_reduce_dim, int64_t* y_dim,
    int64_t* y_reduce_dim) {
  HloInstruction* core_expr = nullptr;
  HloInstruction* reduce_operand = nullptr;
  HloInstruction* reduce_init = nullptr;
  if (!Match(reduce,
             m::Reduce(m::Op(&reduce_operand), m::Constant(&reduce_init)))) {
    return false;
  }
  auto is_pow_based = reduce_operand->opcode() == HloOpcode::kPower;
  auto is_mul_based = reduce_operand->opcode() == HloOpcode::kMultiply &&
                      reduce_operand->operand(0) == reduce_operand->operand(1);
  if (is_mul_based) {
    core_expr = reduce_operand->mutable_operand(0);
  } else if (is_pow_based) {
    HloInstruction* power_const = nullptr;
    if (!Match(reduce_operand,
               m::Power(m::Op(&core_expr),
                        m::Broadcast(m::Constant(&power_const))))) {
      return false;
    }
    if (ShapeUtil::ElementsIn(power_const->shape()) != 1 ||
        !power_const->literal().Get<float>({0}) == 2.0) {
      return false;
    }
  }

  // Check add or sub
  HloInstruction* lhs = nullptr;
  HloInstruction* rhs = nullptr;
  if (Match(core_expr, m::Add(m::Op(&lhs), m::Op(&rhs)))) {
    *is_sub = false;
  } else if (Match(core_expr, m::Subtract(m::Op(&lhs), m::Op(&rhs)))) {
    *is_sub = true;
  } else {
    return false;
  }

  // Check the constants are correct
  if (ShapeUtil::ElementsIn(reduce_init->shape()) != 1 ||
      !reduce_init->literal().IsZero({0})) {
    return false;
  }

  // Check broadcast
  if (!Match(lhs, m::Broadcast(m::Op(x))) ||
      !Match(rhs, m::Broadcast(m::Op(y)))) {
    return false;
  }

  std::stringstream ss;
  ss << "LHS Broadcasting name: " << lhs->name() << "\n";
  ss << "LHS Broadcasting shape: " << lhs->shape() << "\n";
  ss << "LHS Broadcasting dimensions: [";
  for (auto dim: lhs->dimensions()) ss << dim << ", ";
  ss << "]\n";
  LOG(INFO) << ss.str();

  // Check the broadcast + reduce dimensions are correct
  int64_t reduce_dim = reduce->dimensions(0);
  // the reduce dimension must NOT be a broadcasted one
  *x_reduce_dim = -1;
  *y_reduce_dim = -1;
  for (int64_t i = 0; i < lhs->dimensions().size(); i++) {
    if (lhs->dimensions(i) == reduce_dim) {
      *x_reduce_dim = i;
    }
  }
  for (int64_t i = 0; i < rhs->dimensions().size(); i++) {
    if (rhs->dimensions(i) == reduce_dim) {
      *y_reduce_dim = i;
    }
  }
  if (*x_reduce_dim == -1 || *y_reduce_dim == -1) {
    return false;
  }
  // there must be a pair of dims i =/= j such that:
  // x -> i : R, j : B and y -> i : B, j : R
  CHECK(ShapeUtil::Equal(lhs->shape(), rhs->shape()));
  if (lhs->shape().rank() != 3) return false;
  int64_t rank = lhs->shape().rank();
  *x_dim = -1;
  *y_dim = -1;
  for (int64_t i = 0; i < rank; i++) {
    for (int64_t j = 0; j < rank; j++) {
      if (i == j || i == reduce_dim || j == reduce_dim) {
        continue;
      }
      if (absl::c_linear_search(lhs->dimensions(), i) &&
          !absl::c_linear_search(lhs->dimensions(), j) &&
          !absl::c_linear_search(rhs->dimensions(), i) &&
          absl::c_linear_search(rhs->dimensions(), j)) {
        *x_dim = i;
        *y_dim = j;
      }
    }
  }
  if (*x_dim == -1 || *y_dim == -1) {
    return false;
  }

  // TODO: Check reduce computation is add

  return true;
}

Status AlgebraicRewriterVisitor::HandleReduce(HloInstruction* reduce) {
  HloInstruction *x, *y;
  bool is_sub;
  int64_t x_dim, x_reduce_dim, x_dot_dim, y_dim, y_reduce_dim, y_dot_dim;
  if (!MatchDistanceMatrix(reduce, &x, &y, &is_sub, &x_dim, &x_reduce_dim,
                           &y_dim, &y_reduce_dim)) {
    return Status::OK();
  }

  std::stringstream ss;
  ss << ">>> ";
  ss << "Rules match for the euclidean distance matrix. \n";
  ss << "Distance X: " << x->name() << "\n";
  ss << "Distance Y: " << y->name() << "\n";
  ss << "X dimension: " << x_dim << "\n";
  ss << "Y dimension: " << y_dim << "\n";
  ss << "X reduce dimension: " << x_reduce_dim << "\n";
  ss << "Y reduce dimension: " << y_reduce_dim << "\n";
  ss << "Reduce operation: ";
  ss << reduce->name() << "\n";

  // HloComputation* reduce_sum = reduce->called_computations()[0];
  HloComputation* comp = reduce->parent();

  // constants
  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  HloInstruction* two = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2)));
    
  Shape x_shape = x->shape();
  Shape y_shape = y->shape();

  // squares
  HloInstruction* two_x = comp->AddInstruction(
      HloInstruction::CreateBroadcast(x_shape, two, {}));
  HloInstruction* x_squared = comp->AddInstruction(
      HloInstruction::CreateBinary(x_shape, HloOpcode::kPower, x, two_x));
  HloInstruction* two_y = comp->AddInstruction(
      HloInstruction::CreateBroadcast(y_shape, two, {}));
  HloInstruction* y_squared = comp->AddInstruction(
      HloInstruction::CreateBinary(y_shape, HloOpcode::kPower, y, two_y));

  // reduce the squares

  Shape x_reduce_shape = x_squared->shape();
  // x_reduce_shape.DeleteDimension(x_reduce_dim);
  x_reduce_shape.set_dimensions(y_reduce_dim, 1);
  HloInstruction* x_reduce = comp->AddInstruction(HloInstruction::CreateReduce(
      x_reduce_shape, x_squared, zero, {x_reduce_dim}, comp));

  Shape y_reduce_shape = y_squared->shape();
  // y_reduce_shape.DeleteDimension(y_reduce_dim);
  y_reduce_shape.set_dimensions(y_reduce_dim, 1);

  HloInstruction* y_reduce = comp->AddInstruction(HloInstruction::CreateReduce(
      y_reduce_shape, y_squared, zero, {y_reduce_dim}, comp));

  ss << "x_square_shape: " << x_squared->shape() << "\n";
  ss << "y_square_shape: " << y_squared->shape() << "\n";
  ss << "x_reduce_shape: " << x_reduce->shape() << "\n";
  ss << "y_reduce_shape: " << y_reduce->shape() << "\n";
  LOG(INFO) << ss.str();

  // Start construction outer product
  // x y outer product
  auto lhs_reduce_dim = x_reduce_dim;
  auto rhs_reduce_dim = y_reduce_dim;
  HloInstruction* lhs = x;
  HloInstruction* rhs = y;
  if (x_dim < y_dim) {
    lhs_reduce_dim = y_reduce_dim;
    rhs_reduce_dim = x_reduce_dim;
    lhs = y;
    rhs = x;
  } else {
    // failed pre-condition :/
    CHECK((x_dim == 0 && y_dim == 1) || (x_dim == 1 && y_dim == 0));
  }

  Shape outer_shape = lhs->shape();
  Shape rhs_shape = rhs->shape();
  outer_shape.DeleteDimension(lhs_reduce_dim);
  rhs_shape.DeleteDimension(rhs_reduce_dim);
  for (auto i = 0; i < rhs_shape.dimensions_size(); i++) {
    outer_shape.add_dimensions(rhs_shape.dimensions(i));
  }

  PrecisionConfig conf;
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(lhs_reduce_dim);
  dnums.add_rhs_contracting_dimensions(rhs_reduce_dim);
  HloInstruction* outer = comp->AddInstruction(
      HloInstruction::CreateDot(outer_shape, lhs, rhs, dnums, conf));
  
  HloInstruction* x_broadcast = comp->AddInstruction(
      HloInstruction::CreateBroadcast(outer_shape, x_reduce, {x_dim}));
  HloInstruction* y_broadcast = comp->AddInstruction(
      HloInstruction::CreateBroadcast(outer_shape, y_reduce, {y_dim}));

  ss << "Outer product shape: " << outer->shape() << "\n";
  ss << "x broadcast shape: " << x_broadcast->shape() << "\n";
  ss << "y broadcast shape: " << y_broadcast->shape() << "\n";
  LOG(INFO) << ss.str();

  HloInstruction* x_y_sum = comp->AddInstruction(HloInstruction::CreateBinary(
      outer_shape, HloOpcode::kAdd, x_broadcast, y_broadcast));

  ss << "x_y_sum shape: " << x_y_sum->shape() << "\n";
  LOG(INFO) << ss.str();

  HloInstruction* replacement = comp->AddInstruction(
      HloInstruction::CreateBinary(outer_shape, HloOpcode::kAdd, x_y_sum, outer));
  
  ss << "x_y_sum shape: " << x_y_sum->shape() << "\n";
  ss << "replacement shape: " << replacement->shape() << "\n";
  ss << "reduce shape: " << reduce->shape() << "\n";
  LOG(INFO) << ss.str();

  return ReplaceInstruction(reduce, replacement);
}

StatusOr<bool> AlgebraicRewriter::Run(HloModule* module) {
  LOG(INFO) << "Running algebraic rewriter for '" << module->name() << "'";
  AlgebraicRewriterVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla