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

//  A vistor class which traverses graphs and try to sink reshape operations
class ReshapeSinkerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReshapeSinkerVisitor() {}

  //  A helper function to check if the order of an instruiction's dimensions 
  //  matches the index order
  //  Input:
  //    inst: the target instruction
  //  Output:
  //    a bool value to indicates if the instruiction's dimensions matches the index order
  bool DimsAreIndexes(HloInstruction* inst) {
    for (auto i = 0; i < inst->dimensions().size(); i++) {
      if (i != inst->dimensions(i)) {
        return false;
      }
    }
    return true;
  }

  // Get the dimension mapping relationships of the reshape inst and its operand
  // Note: reshape may map one dimension to multiple dimensions or map multiple dimensions to 
  // only one dimensions exp. [6] -> [2,3], [2,2,4]->[2,8]
  //  Input:
  //    reshape: the target reshape instruction
  //    reshape_to_origin: a empty hashtable used to store the dimension mapping relationship from new shape to original shape
  //    origin_to_reshape: a empty hashtable used to store the dimension mapping relationship from original shape to new shape
  //  Output:
  //    a status indicating if the function finds the mapping relationship successfully and the mapping information
  //      would be sotred in the two input hashtables
  Status GetReshapeDimMapping(
      HloInstruction* reshape,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape);

  // A helper function to decides if a dimension is retained by the given reshape instruction
  //  Input:
  //    reshape: the target reshape instruction
  //    reshape_to_origin: a hashtable contains the dimension mapping relationship from new shape to original shape
  //    origin_to_reshape: a hashtable contains the dimension mapping relationship from original shape to new shape
  //  Output:
  //    a bool value to indicates if a dimension is retained by the given reshape instruction
  bool IsReshapeDimUnchanged(
      HloInstruction* reshape, int64_t dim,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& reshape_to_origin,
      absl::flat_hash_map<int64_t, std::vector<int64_t>>& origin_to_reshape);
  // The handler function when visiting a dot instruction
  //  Input:
  //    dot: the currently visted dot instruction
  //  Output:
  //    a bool value to indicates if the processing is successful
  Status HandleDot(HloInstruction* dot) override;
};
}  // namespace

//  Since reshape first flattens an array into a one-dimensional vector of data values, 
//  and then refines this vector into a new shape. So we can find the dimension mapping information
//  by using two pointers to compare dimensions of the original shape and the new shape
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
      // an orginal dimension is splitted to multiple dimesions in the new shape
      int64_t i = reshape_i;
      int64_t cur_reshape_size = reshape->shape().dimensions(reshape_i);
      int64_t cur_operand_size = operand->shape().dimensions(operand_j);
      // the product of dimension sizes of all the new dimesions is equal to
      // the dimension size of the original dimension
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
      // multipled orginal dimensions are merged into a single new dimesion in the new shape
      int64_t j = operand_j;
      int64_t cur_reshape_size = reshape->shape().dimensions(reshape_i);
      int64_t cur_operand_size = operand->shape().dimensions(operand_j);
      //  the dimension size of the the new dimesion is equal to
      //  the product of all dimension sizes of all the original dimensions
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

//  Currently only reshape operations which are direct
//  operands of dot are sunk. The basic logic is once we encounter a dot whose
//  lhs/rhs is a reshape, let's say a reshape's operand is x. We first establish a
//  mapping between the reshape's dimension and x's dimensions, there are 3
//  cases:
//    1. A dimension is retained by the reshape.
//    2. A dimension is split into multiple dimensions by the reshape.
//    3. Multiple dimensions are merged by the reshape.
//  We can only sink reshape to the position below dot if and only if dot's
//  contracting dimension is not new dimension generated by reshape, i.e. the
//  contracting dimension must be a dimension retained by the reshape. If it is
//  the case we can easily sink the reshape to the position below dot and modify
//  corresponding operands and dimensions.
Status ReshapeSinkerVisitor::HandleDot(HloInstruction* dot) {
  std::string prefix = "[ReshapeSinkerVisitor::HandleDot] ";
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  bool rhs_is_lhs = lhs == rhs;
  auto& dnums = dot->dot_dimension_numbers();
  if (rhs_is_lhs && Match(lhs, m::Reshape(m::Op()))) {
    // self-dot
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
        // we can sink reshape iff dot's contracting dimension is not new
        // dimension generated by reshape
        Match(lhs, m::Reshape(m::Op(&new_lhs)));
        new_contracting_dim =
            reshape_to_origin[dnums.lhs_contracting_dimensions()[0]][0];
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
    // lhs != rhs
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
        // lhs is reshape and can be sunk
        Match(lhs, m::Reshape(m::Op(&new_lhs)));
        new_lhs_contracting_dim =
            reshape_to_origin[dnums.lhs_contracting_dimensions()[0]][0];
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
        // rhs is reshape and can be sunk
        Match(rhs, m::Reshape(m::Op(&new_rhs)));
        new_rhs_contracting_dim =
            reshape_to_origin[dnums.rhs_contracting_dimensions()[0]][0];
      } else {
        new_rhs = rhs;
        new_rhs_contracting_dim = dnums.rhs_contracting_dimensions()[0];
      }
    }

    if (lhs != new_lhs || rhs != new_rhs) {
      if (new_lhs->shape().dimensions(new_lhs_contracting_dim) !=
          new_rhs->shape().dimensions(new_rhs_contracting_dim)) {
        // we cannot performin sinking if their contracting_dims don't match after
        // sinking
        LOG(INFO) << prefix << "Skip Dot orig_dot.name=" << dot->name()
                  << " lhs.name=" << lhs->name() << " lhs.contracting_dim="
                  << dnums.lhs_contracting_dimensions()[0]
                  << " lhs.shape=" << lhs->shape().ToString()
                  << " rhs.name=" << rhs->name() << " rhs.contracting_dim="
                  << dnums.lhs_contracting_dimensions()[0];
        return Status::OK();
      }
      Shape orig_dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                                  dot->shape().dimensions());
      DotDimensionNumbers new_dnums;
      new_dnums.add_lhs_contracting_dimensions(new_lhs_contracting_dim);
      new_dnums.add_rhs_contracting_dimensions(new_rhs_contracting_dim);
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