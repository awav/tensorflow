// License TODO ....

#include "tensorflow/compiler/xla/service/dot_order_optimizer.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class DotOrderOptimizerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit DotOrderOptimizerVisitor() {}

  Status HandleDot(HloInstruction* dot) override;
};

}  // namespace

Status DotOrderOptimizerVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // (A B) C => A (B C) if intermediate result is smaller
  HloInstruction *a, *b;
  if (Match(lhs, m::Dot(m::Op(&a), m::Op(&b)))) {
    int64 current_size = ShapeUtil::ElementsIn(lhs->shape());
    int64 rank_a = a->shape().rank();

    int64 lhs_contr_idx =
        dot->dot_dimension_numbers().lhs_contracting_dimensions(
            0);
    int64 rhs_contr_idx =
        dot->dot_dimension_numbers().rhs_contracting_dimensions(
            0);

    int64 proposed_size = 1;

    HloInstruction* inner;
    HloInstruction* outer;

    int64 inner_contr_idx;
    int64 outer_contr_idx;
    int64 inner_with_outer_contr_idx;

    if (lhs_contr_idx < rank_a - 1) {
      // AC
      return Status::OK(); // dimensions won't match (not handled for now; essentially would need to introduce a transpose after)
    } else {
      // BC
      inner = b;
      outer = a;
      inner_contr_idx = lhs_contr_idx - (rank_a - 1);
      outer_contr_idx =
          lhs->dot_dimension_numbers().lhs_contracting_dimensions(0);
      inner_with_outer_contr_idx =
          lhs->dot_dimension_numbers().rhs_contracting_dimensions(0);
    }

    if (inner_contr_idx <= inner_with_outer_contr_idx)
      inner_with_outer_contr_idx--;

    proposed_size *= ShapeUtil::ElementsIn(inner->shape());
    proposed_size /= inner->shape().dimensions(inner_contr_idx);
    proposed_size *= ShapeUtil::ElementsIn(rhs->shape());
    proposed_size /= rhs->shape().dimensions(rhs_contr_idx);

    if (proposed_size < current_size) {
      // inner C
      DotDimensionNumbers inner_dnums;
      inner_dnums.add_lhs_contracting_dimensions(inner_contr_idx);
      inner_dnums.add_rhs_contracting_dimensions(rhs_contr_idx);
      HloInstruction* inner_dot;
      TF_ASSIGN_OR_RETURN(
          inner_dot,
          MakeDotHlo(inner, rhs, inner_dnums, dot->precision_config(),
                     /*preferred_element_type=*/dot->shape().element_type()));

      // outer (inner C)
      DotDimensionNumbers outer_dnums;
      outer_dnums.add_lhs_contracting_dimensions(outer_contr_idx);
      outer_dnums.add_rhs_contracting_dimensions(inner_with_outer_contr_idx);
      HloInstruction* outer_dot;
      TF_ASSIGN_OR_RETURN(
          outer_dot,
          MakeDotHlo(outer, inner_dot, outer_dnums, dot->precision_config(),
                     /*preferred_element_type=*/dot->shape().element_type()));

      return ReplaceInstruction(dot, outer_dot);
    }
  }
  // TODO(dyedgreen): Handle the other case i.e. A(BC) => (AB)C
}

StatusOr<bool> DotOrderOptimizer::Run(HloModule* module) {
  DotOrderOptimizerVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
