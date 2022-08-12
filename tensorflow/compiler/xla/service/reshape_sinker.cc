// License TODO ....

#include "tensorflow/compiler/xla/service/reshape_sinker.h"

#include <stdlib.h>

#include <istream>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

namespace m = match;

class ReshapeSinkerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReshapeSinkerVisitor() {}
  bool DimsAreIndexes(HloInstruction* inst) {
    for (auto i = 0; i < inst->dimensions().size(); i++) {
      if (i != inst->dimensions(i)) {
        return false;
      }
    }
    return true;
  }
  // Get the dim mapping relationships of the reshape inst and its operand
  Status GetReshapeDimMapping(
      HloInstruction* reshape,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape);
  bool IsReshapeDimUnchanged(
      HloInstruction* reshape, int64_t dim,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape);
  Status HandleDot(HloInstruction* dot) override;
};
}  // namespace

Status ReshapeSinkerVisitor::GetReshapeDimMapping(
    HloInstruction* reshape,
    absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
    absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape) {
  HloInstruction* operand;
  std::string prefix = "[ReshapeSinkerVisitor::GetReshapeDimMapping] ";
  Match(reshape, m::Reshape(m::Op(&operand)));
  int64_t reshape_i = 0, operand_j = 0;
  int64_t reshape_dim_end = reshape->shape().dimensions().size() - 1;
  int64_t operand_dim_end = operand->shape().dimensions().size() - 1;
  while (reshape_i <= reshape_dim_end && operand_j <= operand_dim_end) {
    if (reshape->shape().dimensions(reshape_i) ==
        operand->shape().dimensions(operand_j)) {
      // unchanged dim
      reshape_to_origin[reshape_i] = {operand_j};
      origin_to_reshape[operand_j] = {reshape_i};
      reshape_i += 1;
      operand_j += 1;
    } else if (reshape->shape().dimensions(reshape_i) <
               operand->shape().dimensions(operand_j)) {
      int64_t i = reshape_i;
      int64_t cur_reshape_size = reshape->shape().dimensions(reshape_i);
      int64_t cur_operand_size = operand->shape().dimensions(operand_j);
      while (cur_reshape_size < cur_operand_size && i <= reshape_dim_end) {
        i++;
        cur_reshape_size *= reshape->shape().dimensions(i);
      }
      if (cur_reshape_size != cur_operand_size) {
        LOG(FATAL) << prefix << "Reshape Size doesn't match, reshape.name="
                   << reshape->name();
        CHECK(false);
      }
      for (int64_t k = reshape_i; k <= i; ++k) {
        reshape_to_origin[k] = {operand_j};
        if (!origin_to_reshape.contains(operand_j)) {
          origin_to_reshape[operand_j] = {k};
        } else {
          origin_to_reshape[operand_j].push_back(k);
        }
      }
      reshape_i = i + 1;
      operand_j += 1;
    } else if (reshape->shape().dimensions(reshape_i) >
               operand->shape().dimensions(operand_j)) {
      int64_t j = operand_j;
      int64_t cur_reshape_size = reshape->shape().dimensions(reshape_i);
      int64_t cur_operand_size = operand->shape().dimensions(operand_j);
      while (cur_reshape_size > cur_operand_size && j <= operand_dim_end) {
        j++;
        cur_operand_size *= operand->shape().dimensions(j);
      }
      if (cur_reshape_size != cur_operand_size) {
        LOG(FATAL) << prefix << "Reshape Size doesn't match, reshape.name="
                   << reshape->name();
        CHECK(false);
      }
      for (int64_t k = operand_j; k <= j; ++k) {
        origin_to_reshape[k] = {reshape_i};
        if (!reshape_to_origin.contains(reshape_i)) {
          reshape_to_origin[reshape_i] = {k};
        } else {
          reshape_to_origin[reshape_i].push_back(k);
        }
      }
      reshape_i += 1;
      operand_j = j + 1;
    }
  }

  return Status::OK();
}
bool ReshapeSinkerVisitor::IsReshapeDimUnchanged(
    HloInstruction* reshape, int64_t dim,
    absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
    absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape) {
  CHECK(dim < reshape->shape().dimensions().size());
  if (!reshape_to_origin.contains(dim)) {
    LOG(INFO) << "[IsReshapeDimUnchanged] doesn't contain ERROR,reshape.name="
              << reshape->name()
              << " reshape.shape=" << reshape->shape().ToString()
              << " dim=" << dim;
    return false;
  }
  if (reshape_to_origin[dim].size() == 1 &&
      origin_to_reshape[reshape_to_origin[dim][0]].size() == 1) {
    return true;
  }
  return false;
}
Status ReshapeSinkerVisitor::HandleDot(HloInstruction* dot) {
  std::string prefix = "[ReshapeSinkerVisitor::HandleDot] ";
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  bool rhs_is_lhs = lhs == rhs;
  auto& dnums = dot->dot_dimension_numbers();
  if (rhs_is_lhs && Match(lhs, m::Reshape(m::Op()))) {
    if (dnums.lhs_contracting_dimensions().size() == 1 &&
        dnums.lhs_contracting_dimensions()[0] ==
            dnums.rhs_contracting_dimensions()[0]) {
      absl::flat_hash_map<int64_t, std::vector<int64_t>> reshape_to_origin;
      absl::flat_hash_map<int64_t, std::vector<int64_t>> origin_to_reshape;
      TF_RETURN_IF_ERROR(
          GetReshapeDimMapping(lhs, reshape_to_origin, origin_to_reshape));
      HloInstruction* new_lhs = lhs;
      int64_t new_contracting_dim = dnums.lhs_contracting_dimensions()[0];
      if (IsReshapeDimUnchanged(lhs, dnums.lhs_contracting_dimensions()[0],
                                reshape_to_origin, origin_to_reshape)) {
        Match(lhs, m::Reshape(m::Op(&new_lhs)));
        new_contracting_dim =
            reshape_to_origin[dnums.lhs_contracting_dimensions()[0]][0];
        LOG(INFO) << prefix << "orig_self_dot.name=" << dot->name()
                  << " lhs.name=" << lhs->name() << " lhs.contracting_dim="
                  << dnums.lhs_contracting_dimensions()[0]
                  << " lhs.contracting_dim.size="
                  << lhs->shape().dimensions(
                         dnums.lhs_contracting_dimensions()[0])
                  << " lhs.shape=" << lhs->shape().ToString()
                  << " new_lhs.name=" << new_lhs->name()
                  << " new_lhs.contracting_dim=" << new_contracting_dim
                  << " new_lhs.contracting_dim.size="
                  << new_lhs->shape().dimensions(new_contracting_dim)
                  << " new_lhs.shape=" << new_lhs->shape().ToString();
      } else {
        return Status::OK();
      }
      if (new_lhs != lhs) {
        LOG(INFO) << prefix
                  << "Start Replace SelfDot orig_dot.name=" << dot->name()
                  << " lhs.name=" << lhs->name()
                  << " new_lhs.name=" << new_lhs->name();
        Shape orig_dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                                    dot->shape().dimensions());
        DotDimensionNumbers new_dnums;
        new_dnums.add_lhs_contracting_dimensions(new_contracting_dim);
        new_dnums.add_rhs_contracting_dimensions(new_contracting_dim);
        // TF_ASSIGN_OR_RETURN(const Shape* new_lhs_shape,
        // GetShapePtr(new_lhs)); TF_ASSIGN_OR_RETURN(const Shape*
        // new_rhs_shape, GetShapePtr(new_lhs));
        TF_ASSIGN_OR_RETURN(Shape new_dot_shape,
                            ShapeInference::InferDotOpShape(
                                new_lhs->shape(), new_lhs->shape(), new_dnums,
                                dot->shape().element_type()));
        HloInstruction* new_dot = dot->parent()->AddInstruction(
            HloInstruction::CreateDot(new_dot_shape, new_lhs, new_lhs,
                                      new_dnums, dot->precision_config()));
        HloInstruction* new_reshape = dot->parent()->AddInstruction(
            HloInstruction::CreateReshape(orig_dot_shape, new_dot));

        LOG(INFO) << prefix << "SelfDot orig_dot.name=" << dot->name()
                  << " orig_dot.shape=" << dot->shape().ToString()
                  << "; new_dot.shape=" << new_dot->shape().ToString()
                  << " new_dot.lhs/rhs.name=" << new_lhs->name()
                  << " new_dot.lhs/rhs.shape=" << new_lhs->shape().ToString()
                  << " new_reshape.shape=" << new_reshape->shape().ToString();
        return ReplaceInstruction(dot, new_reshape);
      }
    }

  } else if (dnums.lhs_contracting_dimensions().size() == 1 &&
             dnums.rhs_contracting_dimensions().size() == 1) {
    HloInstruction *new_lhs = lhs, *new_rhs = rhs;
    int64_t new_lhs_contracting_dim = dnums.lhs_contracting_dimensions()[0],
            new_rhs_contracting_dim = dnums.rhs_contracting_dimensions()[0];
    if (Match(lhs, m::Reshape(m::Op()))) {
      absl::flat_hash_map<int64_t, std::vector<int64_t>> reshape_to_origin;
      absl::flat_hash_map<int64_t, std::vector<int64_t>> origin_to_reshape;
      TF_RETURN_IF_ERROR(
          GetReshapeDimMapping(lhs, reshape_to_origin, origin_to_reshape));
      if (IsReshapeDimUnchanged(lhs, dnums.lhs_contracting_dimensions()[0],
                                reshape_to_origin, origin_to_reshape)) {
        Match(lhs, m::Reshape(m::Op(&new_lhs)));
        new_lhs_contracting_dim =
            reshape_to_origin[dnums.lhs_contracting_dimensions()[0]][0];
        LOG(INFO) << prefix << "lhs: orig_dot.name=" << dot->name()
                  << " lhs.name=" << lhs->name() << " lhs.contracting_dim="
                  << dnums.lhs_contracting_dimensions()[0]
                  << " lhs.contracting_dim.size="
                  << lhs->shape().dimensions(
                         dnums.lhs_contracting_dimensions()[0])
                  << " lhs.shape=" << lhs->shape().ToString()
                  << " new_lhs.name=" << new_lhs->name()
                  << " new_lhs.contracting_dim=" << new_lhs_contracting_dim
                  << " new_lhs.contracting_dim.size="
                  << new_lhs->shape().dimensions(new_lhs_contracting_dim)
                  << " new_lhs.shape=" << new_lhs->shape().ToString();
      } else {
        new_lhs = lhs;
        new_lhs_contracting_dim = dnums.lhs_contracting_dimensions()[0];
      }
    }
    if (Match(rhs, m::Reshape(m::Op()))) {
      absl::flat_hash_map<int64_t, std::vector<int64_t>> reshape_to_origin;
      absl::flat_hash_map<int64_t, std::vector<int64_t>> origin_to_reshape;
      TF_RETURN_IF_ERROR(
          GetReshapeDimMapping(rhs, reshape_to_origin, origin_to_reshape));
      if (IsReshapeDimUnchanged(rhs, dnums.rhs_contracting_dimensions()[0],
                                reshape_to_origin, origin_to_reshape)) {
        Match(rhs, m::Reshape(m::Op(&new_rhs)));
        new_rhs_contracting_dim =
            reshape_to_origin[dnums.rhs_contracting_dimensions()[0]][0];
        LOG(INFO) << prefix << "rhs: orig_dot.name=" << dot->name()
                  << " rhs.name=" << rhs->name() << " rhs.contracting_dim="
                  << dnums.rhs_contracting_dimensions()[0]
                  << " lhs.contracting_dim.size="
                  << rhs->shape().dimensions(
                         dnums.rhs_contracting_dimensions()[0])
                  << " rhs.shape=" << rhs->shape().ToString()
                  << " new_rhs.name=" << new_rhs->name()
                  << " new_rhs.contracting_dim=" << new_rhs_contracting_dim
                  << " new_rhs.contracting_dim.size="
                  << new_rhs->shape().dimensions(new_rhs_contracting_dim)
                  << " new_rhs.shape=" << new_rhs->shape().ToString();
      } else {
        new_rhs = rhs;
        new_rhs_contracting_dim = dnums.rhs_contracting_dimensions()[0];
      }
    }

    if (lhs != new_lhs || rhs != new_rhs) {
      if (new_lhs->shape().dimensions(new_lhs_contracting_dim) !=
          new_rhs->shape().dimensions(new_rhs_contracting_dim)) {
        LOG(INFO)
            << prefix << "Skip Dot orig_dot.name=" << dot->name()
            << " lhs.name=" << lhs->name()
            << " lhs.contracting_dim=" << dnums.lhs_contracting_dimensions()[0]
            << " lhs.contracting_dim.size="
            << lhs->shape().dimensions(dnums.lhs_contracting_dimensions()[0])
            << " lhs.shape=" << lhs->shape().ToString()
            << " new_lhs.name=" << new_lhs->name()
            << " new_lhs.contracting_dim=" << new_lhs_contracting_dim
            << " new_lhs.contracting_dim.size="
            << new_lhs->shape().dimensions(new_lhs_contracting_dim)
            << " new_lhs.shape=" << new_lhs->shape().ToString()
            << " rhs.name=" << rhs->name()
            << " rhs.contracting_dim=" << dnums.lhs_contracting_dimensions()[0]
            << " rhs.contracting_dim.size="
            << rhs->shape().dimensions(dnums.rhs_contracting_dimensions()[0])
            << " rhs.shape=" << rhs->shape().ToString()
            << " new_rhs.name=" << new_rhs->name()
            << " new_rhs.contracting_dim=" << new_rhs_contracting_dim
            << " new_rhs.contracting_dim.size="
            << new_rhs->shape().dimensions(new_rhs_contracting_dim)
            << " new_rhs.shape=" << new_rhs->shape().ToString();
        return Status::OK();
      }
      LOG(INFO) << prefix << "Start Replace Dot orig_dot.name=" << dot->name()
                << " lhs.name=" << lhs->name()
                << " new_lhs.name=" << new_lhs->name()
                << " rhs.name=" << rhs->name()
                << " new_rhs.name=" << new_rhs->name();
      Shape orig_dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                                  dot->shape().dimensions());
      DotDimensionNumbers new_dnums;
      new_dnums.add_lhs_contracting_dimensions(new_lhs_contracting_dim);
      new_dnums.add_rhs_contracting_dimensions(new_rhs_contracting_dim);
      // TF_ASSIGN_OR_RETURN(const Shape* new_lhs_shape, GetShapePtr(new_lhs));
      // TF_ASSIGN_OR_RETURN(const Shape* new_rhs_shape, GetShapePtr(new_rhs));
      TF_ASSIGN_OR_RETURN(Shape new_dot_shape,
                          ShapeInference::InferDotOpShape(
                              new_lhs->shape(), new_rhs->shape(), new_dnums,
                              dot->shape().element_type()));
      HloInstruction* new_dot = dot->parent()->AddInstruction(
          HloInstruction::CreateDot(new_dot_shape, new_lhs, new_rhs, new_dnums,
                                    dot->precision_config()));
      HloInstruction* new_reshape = dot->parent()->AddInstruction(
          HloInstruction::CreateReshape(orig_dot_shape, new_dot));

      LOG(INFO) << prefix << "Dot orig_dot.name=" << dot->name()
                << " orig_dot.shape=" << dot->shape().ToString()
                << "; new_dot.shape=" << new_dot->shape().ToString()
                << " new_dot.lhs.name=" << new_lhs->name()
                << " new_dot.lhs.shape=" << new_lhs->shape().ToString()
                << " new_dot.rhs.name=" << new_rhs->name()
                << " new_dot.rhs.shape=" << new_rhs->shape().ToString()
                << " new_reshape.shape=" << new_reshape->shape().ToString();
      return ReplaceInstruction(dot, new_reshape);
    }
  }

  return Status::OK();
}

StatusOr<bool> ReshapeSinker::Run(HloModule* module) {
  ReshapeSinkerVisitor visitor;
  LOG(INFO) << "Running ReshapeSinker for " << module->name() << "'";
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto change, visitor.RunOnModule(module));
  changed |= change;
  return changed;
}

}  // namespace xla