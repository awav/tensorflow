
#include "tensorflow/compiler/xla/service/do_random_shit.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include <stdio.h>

namespace xla {

namespace {

namespace m = match;

class FirstTryVisitor : public DfsHloRewriteVisitor {
  public:
    explicit FirstTryVisitor() {}

    Status HandleDot(HloInstruction* dot) override;

    // bool Run(HloComputation* computation);

  // it might blow up if we don't do this, or it might be fine ?
  // private:
  //   HloComputation* computation_;
};

} // namespace

// void AlgebraicSimplifierVisitor::ResetState(HloComputation* computation) {
//   changed_ = false;
//   ResetVisitStates();
//   computation_ = computation;
// }

// bool AlgebraicSimplifierVisitor::Run(HloComputation* computation) {
//   ResetState(computation);
//   TF_CHECK_OK(computation->Accept(this));
//   return changed_ || changed();
// }

Status FirstTryVisitor::HandleDot(HloInstruction* dot) {
  // CHECK(computation_ == dot->parent()); // not sure what this does ...
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  printf("Visiting a handle dot ...\n");

  // (A B) C => A (B C), C = rhs
  HloInstruction *a, *b;
  if (
    Match(lhs, m::Dot(m::Op(&a), m::Op(&b)))
  ) {
    printf("We had a match...\n");

    int64 current_size = ShapeUtil::ElementsIn(lhs->shape());

    // determine if the other option is to contract BC or AC
    int64 rank_a = a->shape().rank();
    // int64 rank_b = b->shape().rank();
    int64 lhs_contr_idx = dot->dot_dimension_numbers().lhs_contracting_dimensions(0); // does this need to support multi dim?
    int64 rhs_contr_idx = dot->dot_dimension_numbers().rhs_contracting_dimensions(0); // does this need to support multi dim?

    int64 proposed_size = 1;

    HloInstruction* inner;
    HloInstruction* outer;

    int64 inner_contr_idx;
    int64 outer_contr_idx;
    int64 inner_with_outer_contr_idx;

    if (lhs_contr_idx < rank_a) {
      // AC
      inner = a;
      outer = b;
      inner_contr_idx = lhs_contr_idx;
      outer_contr_idx = lhs->dot_dimension_numbers().rhs_contracting_dimensions(0);

      inner_with_outer_contr_idx = lhs->dot_dimension_numbers().lhs_contracting_dimensions(0);
    } else {
      // BC
      inner = b;
      outer = a;
      inner_contr_idx = lhs_contr_idx - rank_a;
      outer_contr_idx = lhs->dot_dimension_numbers().lhs_contracting_dimensions(0);

      inner_with_outer_contr_idx = lhs->dot_dimension_numbers().rhs_contracting_dimensions(0);
    }

    if (inner_contr_idx < inner_with_outer_contr_idx)
      inner_with_outer_contr_idx --;

    proposed_size *= ShapeUtil::ElementsIn(inner->shape());
    proposed_size /= inner->shape().dimensions(inner_contr_idx);
    proposed_size *= ShapeUtil::ElementsIn(rhs->shape());
    proposed_size /= rhs->shape().dimensions(rhs_contr_idx);

    printf("Current intermediate size: %d\n", current_size);
    printf("Proposed intermediate size: %d\n", proposed_size);

    if (proposed_size < current_size) {
      // inner C
      DotDimensionNumbers inner_dnums;
      inner_dnums.add_lhs_contracting_dimensions(inner_contr_idx);
      inner_dnums.add_rhs_contracting_dimensions(rhs_contr_idx);
      // for (
      //   int64 batch_dim = 0;
      //   batch_dim < dot->dot_dimension_numbers().lhs_batch_dimensions_size();
      //   ++batch_dim
      // ) {
      //   inner_dnums.add_rhs_batch_dimensions(
      //       dot->dot_dimension_numbers().rhs_batch_dimensions(batch_dim));
      //   inner_dnums.add_lhs_batch_dimensions(
      //       dot->dot_dimension_numbers().lhs_batch_dimensions(batch_dim));
      // }
      HloInstruction* inner_dot;
      TF_ASSIGN_OR_RETURN(inner_dot, MakeDotHlo(
        inner, rhs, inner_dnums, dot->precision_config(), /*preferred_element_type=*/dot->shape().element_type()));
      
      // outer (inner C)
      DotDimensionNumbers outer_dnums;
      outer_dnums.add_lhs_contracting_dimensions(outer_contr_idx);
      outer_dnums.add_rhs_contracting_dimensions(inner_with_outer_contr_idx);
      HloInstruction* outer_dot;
      TF_ASSIGN_OR_RETURN(outer_dot, MakeDotHlo(
        outer, inner_dot, outer_dnums, dot->precision_config(), /*preferred_element_type=*/dot->shape().element_type()));

      return ReplaceInstruction(dot, outer_dot);
    }
  }
}

// } // namespace

StatusOr<bool> RandomShitPass::Run(HloModule* module) {
  // if (!module->has_entry_computation()) {
  //   printf("DELETE ME: no entry computatio\n");
  //   return false;
  // }

  // // we probably want to run on all computations
  // // or at least all "NonfusionComputations"?
  // HloComputation* entry = module->entry_computation();

  printf("Running the RunOnModule...\n");
  FirstTryVisitor visitor;
  return visitor.RunOnModule(module);

  // HloInstruction* root = entry->root_instruction();
  // if (!root) {
  //   printf("DELETE ME: No root instruction\n");
  //   return false;
  // }
  // printf("Root OptCode: %s\n", HloOpcodeString(root->opcode()).c_str());
  // return false;
}

} // namespace xla
