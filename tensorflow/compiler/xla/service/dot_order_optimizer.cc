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

  // TODO: Abort if multi index dot (which are currently not supported by XLA in
  // general)

  HloInstruction *a, *b, *c;

  // (A B) C => A (B C) if intermediate result is smaller
  if (Match(lhs, m::Dot(m::Op(&a), m::Op(&b)))) {
    /*
      We want to rewrite (AB)C -> A(BC)
      =================================
      Consider possible indexes involved:

         A         | B         | C
         ----------+-----------+-------
      1) 0 ab    n | 0 ba bc m | 0 cb l
      2) 0 ab    n | 0 bc ba m | 0 cb l
      3) 0 ab ac n | 0 ba    m | 0 ca l
      4) 0 ac ab n | 0 ba    m | 0 ca l

      1) => ab; ba; abc = bc + rank(a) - 2; cab = cb;
      2) => ab; ba; abc = bc + rank(a) - 1; cab = cb;
      3) => ab; ba; abc = ac - 1; cab = ca;
      4) => ab; ba; abc = ac; cab = ca;

      1) bc > 0 -> abc >= rank(a) - 1
      2) abc >= rank(a) - 1
      3) abc < ac && ac < rank(a) -> abc < rank(a) - 1
      4) ac < ab && abc = ac && ab < rank(a) -> abc < rank(a) - 1

      ==> can distinquish cases 1/2 vs 3/4 as abc < rank(a) - 1 VS >= rank(a)-1
      ==> case 3, 4 will change overall index order if flipped, so would
          require an additional transpose; skip them for now
    */
    c = rhs;

    int64 rank_a = a->shape().rank();

    int64 contr_ab_c =
        dot->dot_dimension_numbers().lhs_contracting_dimensions(0);

    if (contr_ab_c >= rank_a - 1) {
      // Case 1 or 2, three indices are stright forward
      int64 contr_a_b =
          lhs->dot_dimension_numbers().lhs_contracting_dimensions(0);
      int64 contr_b_a =
          lhs->dot_dimension_numbers().rhs_contracting_dimensions(0);
      int64 contr_c_b =
          dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
      // If the bc index falls onto or grater than ba, increase it
      int64 contr_b_c =
          dot->dot_dimension_numbers().rhs_contracting_dimensions(0) -
          (rank_a - 1);
      if (contr_b_c >= contr_b_a) contr_b_c += 1;

      // TODO: Create new inner dot and new outer dot ...
    }

    int64 current_size = ShapeUtil::ElementsIn(lhs->shape());

    int64 lhs_contr_idx =
        dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
    int64 rhs_contr_idx =
        dot->dot_dimension_numbers().rhs_contracting_dimensions(0);

    int64 proposed_size = 1;

    HloInstruction* inner;
    HloInstruction* outer;

    int64 inner_contr_idx;
    int64 outer_contr_idx;
    int64 inner_with_outer_contr_idx;

    if (lhs_contr_idx < rank_a - 1) {
      // AC
      return Status::OK();  // dimensions won't match (not handled for now;
                            // essentially would need to introduce a transpose
                            // after)
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
