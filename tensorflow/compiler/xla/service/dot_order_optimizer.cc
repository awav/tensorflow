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

         A         | B         | C
      -------------+-----------+-------
      1) 0 ab    n | 0 ba bc m | 0 cb l
      2) 0 ab    n | 0 bc ba m | 0 cb l
      3) 0 ab ac n | 0 ba    m | 0 ca l
      4) 0 ac ab n | 0 ba    m | 0 ca l

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
          dot->dot_dimension_numbers().lhs_contracting_dimensions(0) -
          (rank_a - 1);
      if (contr_b_c >= contr_b_a) contr_b_c += 1;

      DotDimensionNumbers inner_dnums;
      inner_dnums.add_lhs_contracting_dimensions(contr_b_c);
      inner_dnums.add_rhs_contracting_dimensions(contr_c_b);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * inner,
          MakeDotHlo(b, c, inner_dnums, dot->precision_config(), dot->shape().element_type()));

      int64 contr_bc_a = contr_b_a < contr_b_c ? contr_b_a : contr_b_a - 1;
      int64 contr_a_bc = contr_a_b;

      DotDimensionNumbers outer_dnums;
      outer_dnums.add_lhs_contracting_dimensions(contr_a_bc);
      outer_dnums.add_rhs_contracting_dimensions(contr_bc_a);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * outer,
          MakeDotHlo(a, inner, outer_dnums, dot->precision_config(), dot->shape().element_type()));

      return ReplaceInstruction(dot, outer);
    }
  }
  // TODO(dyedgreen): Handle the other case i.e. A(BC) => (AB)C
}

StatusOr<bool> DotOrderOptimizer::Run(HloModule* module) {
  DotOrderOptimizerVisitor visitor;
  return visitor.RunOnModule(module);
}

}  // namespace xla
