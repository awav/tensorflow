// License TODO ....
#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

#include <stdlib.h>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class IntermediateTensorSplitterRewriteVisitor : public DfsHloRewriteVisitor {
  int64 max_intermediate_bytes;
  int64 target_intermediate_bytes;
  HloModule* parent_module;

 public:
  explicit IntermediateTensorSplitterRewriteVisitor(
      int64 max_intermediate_bytes, int64 target_intermediate_bytes,
      HloModule* parent_module)
      : max_intermediate_bytes(max_intermediate_bytes),
        target_intermediate_bytes(target_intermediate_bytes),
        parent_module(parent_module) {}

  // Determine if an operand is large enough such that we are
  // interested in splitting it.
  bool OperandShouldBeSplit(HloInstruction* inst);

  // Determine if an operand can be split by traversing it's
  // inputs until a splittable node is found. This will also
  // return a list leafs and a list of dimensions which can
  // not be split (if an internal op is only partially point-
  // wise).
  bool OperandCanBeSplit(HloInstruction* inst,
                         std::vector<HloInstruction*>* split_leafs = nullptr,
                         std::vector<int64>* original_dimensions = nullptr,
                         std::vector<int64>* exclude_dimensions = nullptr);

  // Matches any pointwise unary operator which has no side effects.
  static bool MatchPointwiseUnary(HloInstruction* inst,
                                  HloInstruction** operand = nullptr);

  // Matches any pointwise n-ary operator.
  static bool MatchPointwiseNary(
      HloInstruction* inst, std::vector<HloInstruction*>* operands = nullptr);

  // Matches a reduce operation where all operands have the same shape
  // and all initilizers are scalars.
  static bool MatchSupportedReduce(HloInstruction* inst);

  static bool MatchSupportedNestedReduce(HloInstruction* inst);

  // Determine the best dimesion to split on, excluding a given one.
  int64 BestSplitDim(HloInstruction* inst, absl::Span<const int64> excluded);

  // Given a split dimension, determine the best possible split
  // size. If no split size is possible, returns -1.
  int64 BestSplitSize(HloInstruction* inst, int64 split_dim);

  Status HandleDot(HloInstruction* dot) override;

  Status HandleReduce(HloInstruction* reduce) override;

  // Collect computation for the instruction we want to split
  // and split the parameters. The parameters are returned pre-
  // split such that they can be used verbatim inside a call.
  // The returned instruction is the root instruction of the
  // computation.
  class Splitter {
    HloInstruction* param_;   // single tuple param instruction
    HloInstruction* offset_;  // get offset from tuple param
    std::vector<HloInstruction*>
        parameters_;  // initialize tuple parameter elements

    HloComputation::Builder& builder_;
    absl::Span<HloInstruction*> leafs_;

   public:
    explicit Splitter(HloComputation::Builder& builder, HloComputation* parent,
                      absl::Span<HloInstruction*> leafs)
        : builder_(builder), leafs_(leafs) {
      // Create the offset init (0)
      HloInstruction* init_offset = parent->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(0)));
      parameters_.push_back(init_offset);

      // Make a param, the shape can be added to over time to get correct shape
      Shape param_shape = ShapeUtil::MakeTupleShape({init_offset->shape()});
      param_ = builder.AddInstruction(
          HloInstruction::CreateParameter(0, param_shape, "loop_param"));
      offset_ = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          init_offset->shape(), param_, 0));
    }

    StatusOr<HloInstruction*> SplitInstruction(HloInstruction* inst,
                                               int64 split_dim,
                                               int64 split_size);

    StatusOr<HloInstruction*> SplitLeafDot(HloInstruction* dot, int64 split_dim,
                                           int64 split_size);

    StatusOr<HloInstruction*> SplitLeafBroadcast(HloInstruction* broadcast,
                                                 int64 split_dim,
                                                 int64 split_size);

    StatusOr<HloInstruction*> SplitLeafIota(HloInstruction* iota,
                                            int64 split_dim, int64 split_size);

    // Add the parameter and returnd it's index in the tuple. If get_tuple
    // is passed, it will also create an accessor for the parameter.
    int64 AddParameter(HloInstruction* inst,
                       HloInstruction** get_tuple = nullptr) {
      int64 idx = parameters_size();
      parameters_.push_back(inst);
      param_->mutable_shape()->mutable_tuple_shapes()->push_back(inst->shape());
      if (get_tuple != nullptr) {
        *get_tuple = builder_.AddInstruction(
            HloInstruction::CreateGetTupleElement(inst->shape(), param_, idx));
      }
      return idx;
    }

    // Generates the final output tuple from the given root
    // computation part.
    HloInstruction* BuildOutputTuple(int64 split_dim, int64 split_size,
                                     HloInstruction* original,
                                     HloInstruction* part,
                                     bool combine_with_sum = false);

    int64 parameters_size() { return parameters_.size(); }

    HloInstruction* parameters(int64 idx) { return parameters_.at(idx); }

    std::vector<HloInstruction*>& parameters() { return parameters_; }

    HloInstruction* offset() { return offset_; }
  };
};

}  // namespace

bool IntermediateTensorSplitterRewriteVisitor::OperandShouldBeSplit(
    HloInstruction* inst) {
  if (!inst->shape().IsArray()) return false;
  return ShapeUtil::ByteSizeOfElements(inst->shape()) > max_intermediate_bytes;
}

bool IntermediateTensorSplitterRewriteVisitor::OperandCanBeSplit(
    HloInstruction* inst, std::vector<HloInstruction*>* split_leafs,
    std::vector<int64>* original_dimensions,
    std::vector<int64>* exclude_dimensions) {
  HloInstruction* next;
  std::vector<HloInstruction*> next_vec;
  if (Match(inst, m::Dot(m::Op(), m::Op()))) {
    // Base case: A Dot produces this large intermediate tensor
    if (split_leafs != nullptr) split_leafs->push_back(inst);
    return true;
  } else if (Match(inst, m::Broadcast(m::Op()))) {
    // Base case: A broadcast can be split
    if (split_leafs != nullptr) split_leafs->push_back(inst);
    return true;
  } else if (Match(inst, m::Iota())) {
    // Base case: An Iota can be split!
    if (split_leafs != nullptr) split_leafs->push_back(inst);
    return true;
  } else if (Match(inst, m::Transpose(m::Op(&next)))) {
    // A transpose changes the dimensions, so we need to
    // update the original_dimensions array.
    if (original_dimensions != nullptr) {
      std::vector<int64> old_original_dimensions(original_dimensions->begin(),
                                                 original_dimensions->end());
      for (int64 i = 0; i < original_dimensions->size(); i++) {
        (*original_dimensions)[i] =
            old_original_dimensions[inst->dimensions(i)];
      }
    }
    return OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions);
  } else if (MatchSupportedNestedReduce(inst)) {
    return OperandCanBeSplit(inst->mutable_operand(0), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
    // We can split a triangular solve on some (but not all)
    // dims
    if (original_dimensions != nullptr && exclude_dimensions != nullptr) {
      if (inst->triangular_solve_options().left_side()) {
        // exclude second to last : Ax = y
        exclude_dimensions->push_back(
            (*original_dimensions)[original_dimensions->size() - 2]);
      } else {
        // exclude last : xA = y
        exclude_dimensions->push_back(
            (*original_dimensions)[original_dimensions->size() - 1]);
      }
    }
    // We can't split the matrix for now, so ignore it
    return OperandCanBeSplit(inst->mutable_operand(1), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (MatchPointwiseUnary(inst, &next)) {
    // This is a special case seperate from nary,
    // since we can make it tail recursive :)
    return OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions);
  } else if (MatchPointwiseNary(inst, &next_vec)) {
    for (HloInstruction* next : next_vec) {
      // this path is not tail recursive :(
      if (!OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions))
        return false;
    }
    return true;
  } else {
    return false;
  }
}

bool IntermediateTensorSplitterRewriteVisitor::MatchPointwiseUnary(
    HloInstruction* inst, HloInstruction** operand) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() == 1) {
    if (operand != nullptr) {
      *operand = inst->mutable_operand(0);
    }
    return true;
  } else {
    return false;
  }
}

bool IntermediateTensorSplitterRewriteVisitor::MatchPointwiseNary(
    HloInstruction* inst, std::vector<HloInstruction*>* operands) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() > 0) {
    if (operands != nullptr) {
      for (int64 i = 0; i < inst->operand_count(); i++)
        operands->push_back(inst->mutable_operand(i));
    }
    return true;
  } else {
    return false;
  }
}

bool IntermediateTensorSplitterRewriteVisitor::MatchSupportedReduce(
    HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReduce) {
    int64 opt_count = inst->operand_count() / 2;
    if (opt_count < 1) return false;

    for (int64 i = 1; i < opt_count; i++)
      if (!ShapeUtil::EqualIgnoringElementType(inst->operand(0)->shape(),
                                               inst->operand(i)->shape()))
        return false;

    for (int64 i = 0; i < opt_count; i++)
      if (!ShapeUtil::IsScalar(inst->operand(opt_count + i)->shape()))
        return false;

    return true;
  } else {
    return false;
  }
}

bool IntermediateTensorSplitterRewriteVisitor::MatchSupportedNestedReduce(
    HloInstruction* inst) {
  return MatchSupportedReduce(inst) && inst->operand_count() == 2;
}

int64 IntermediateTensorSplitterRewriteVisitor::BestSplitDim(
    HloInstruction* inst, absl::Span<const int64> excluded) {
  const Shape& shape = inst->shape();
  int64 best_dim = -1, best_split = 0;  // ShapeUtil::ElementsIn(inst->shape());
  for (int64 i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) continue;
    int64 split = BestSplitSize(inst, i);
    if (split == -1 || split <= best_split) continue;
    best_split = split;
    best_dim = i;
  }
  return best_dim;
}

const int64 primes[64] = {2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,
                          37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,
                          83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137,
                          139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
                          197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                          263, 269, 271, 277, 281, 283, 293, 307, 311};

int64 BestSplitSizeFold(int64 (&factors)[64], int offset, int64 current,
                        int64 best, int64 size, int64 max_size) {
  if (offset >= 64) {
    return best;
  } else {
    if (factors[offset] > 0) {
      int64 current_prime = primes[offset] * current;
      if (size / current_prime <= max_size && current_prime < best) {
        best = current_prime;
      }
      factors[offset]--;
      best = BestSplitSizeFold(factors, offset, current_prime, best, size,
                               max_size);
      factors[offset]++;
    }
    return BestSplitSizeFold(factors, offset + 1, current, best, size,
                             max_size);
  }
}

int64 IntermediateTensorSplitterRewriteVisitor::BestSplitSize(
    HloInstruction* inst, int64 split_dim) {
  // find list of prime factors
  int64 factors[64];
  int64 tmp_size = inst->shape().dimensions(split_dim);
  for (int i = 0; i < 64; i++) {
    factors[i] = 0;
    while (tmp_size % primes[i] == 0) {
      factors[i]++;
      tmp_size /= primes[i];
    }
  }

  int64 size = inst->shape().dimensions(split_dim);
  int64 full_size_bytes =
      ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type()) *
      ShapeUtil::ElementsIn(inst->shape());
  int64 max_size = max_intermediate_bytes * size / full_size_bytes;
  int64 factor = BestSplitSizeFold(factors, 0, 1, size, size, max_size);
  return size / factor;
}

StatusOr<HloInstruction*>
IntermediateTensorSplitterRewriteVisitor::Splitter::SplitInstruction(
    HloInstruction* inst, int64 split_dim, int64 split_size) {
  if (absl::c_linear_search(leafs_, inst)) {
    if (Match(inst, m::Dot())) {
      return SplitLeafDot(inst, split_dim, split_size);
    } else if (Match(inst, m::Broadcast())) {
      return SplitLeafBroadcast(inst, split_dim, split_size);
    } else if (Match(inst, m::Iota())) {
      return SplitLeafIota(inst, split_dim, split_size);
    }
  } else {
    HloInstruction* operand;
    std::vector<HloInstruction*> operands;

    if (Match(inst, m::Transpose(m::Op(&operand)))) {
      // For a transpose, the transpose might change which dimension is
      // being split. So we obtain the new split dimension and then
      // recursively a new operand to make a clone.
      int64 operand_split_dim = inst->dimensions(split_dim);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(operand, operand_split_dim, split_size));
      std::vector<HloInstruction*> ops = {new_operand};
      return builder_.AddInstruction(inst->CloneWithNewOperands(
          new_operand->shape(), absl::MakeSpan(ops)));
    } else if (MatchSupportedNestedReduce(inst)) {
      // For a reduce, split the 0th and only operand
      // (the initializer a scalar, so all we need to do
      // is update the shape and clone the operand with new
      // inputs)
      int64 operand_split_dim = split_dim;  // split dim in operand
      if (inst->dimensions(0) <= split_dim) operand_split_dim += 1;

      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                          SplitInstruction(inst->mutable_operand(0),
                                           operand_split_dim, split_size));

      HloInstruction* init_operand = inst->mutable_operand(1);
      HloInstruction* new_init_operand;
      AddParameter(init_operand, &new_init_operand);

      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      return builder_.AddInstruction(inst->CloneWithNewOperands(
          new_shape, {new_operand, new_init_operand}));
    } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(inst->mutable_operand(1), split_dim, split_size));
      HloInstruction* mat;
      HloInstruction* split_op_param;
      AddParameter(inst->mutable_operand(0), &mat);
      return builder_.AddInstruction(
          inst->CloneWithNewOperands(new_operand->shape(), {mat, new_operand}));
    } else if (MatchPointwiseNary(inst, &operands)) {
      // For a pointwise operation recursively obtain the new operands and
      // clone the operation.
      std::vector<HloInstruction*> ops;
      for (HloInstruction* operand : operands) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                            SplitInstruction(operand, split_dim, split_size));
        ops.push_back(new_operand);
      }
      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      return builder_.AddInstruction(
          inst->CloneWithNewOperands(new_shape, absl::MakeSpan(ops)));
    } else {
      // Invariant violation
      // TODO: Is there a more idiomatic way to return a bad status?
      LOG(ERROR) << "Trying to split invalid operation.";
      CHECK(false);
    }
  }
}

StatusOr<HloInstruction*>
IntermediateTensorSplitterRewriteVisitor::Splitter::SplitLeafDot(
    HloInstruction* dot, int64 split_dim, int64 split_size) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // For the dot we identify the parameter to split and then
  // Generate the final dot operation, as well as the operand
  // vector.

  Shape dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                         dot->shape().dimensions());
  dot_shape.set_dimensions(split_dim, split_size);

  auto& dnums = dot->dot_dimension_numbers();
  int64 dims_lhs =
      lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();

  HloInstruction *split_op, *join_op;
  bool split_is_lhs;
  if (split_dim < dims_lhs) {
    // We are splitting up the lhs
    split_is_lhs = true;
    split_op = lhs;
    join_op = rhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64 i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.lhs_contracting_dimensions(i)) split_dim += 1;
    }
  } else {
    // We are splitting up the rhs
    split_is_lhs = false;
    split_dim -= dims_lhs;
    split_op = rhs;
    join_op = lhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64 i = 0; i < dnums.rhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.rhs_contracting_dimensions(i)) split_dim += 1;
    }
  }

  // add parameters
  HloInstruction* split_op_param;
  int64 split_op_tuple_idx = AddParameter(split_op, &split_op_param);
  HloInstruction* join_op_param;
  int64 join_op_tuple_idx = AddParameter(join_op, &join_op_param);

  // dynamic slice by index
  Shape split_shape = ShapeUtil::MakeShape(split_op->shape().element_type(),
                                           split_op->shape().dimensions());
  split_shape.set_dimensions(split_dim, split_size);

  std::vector<HloInstruction*> start_indices;
  for (int64 dim = 0; dim < split_shape.dimensions_size(); dim++) {
    if (dim == split_dim) {
      start_indices.push_back(offset_);
    } else {
      start_indices.push_back(builder_.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(0))));
    }
  }
  HloInstruction* split_slice =
      builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
          split_shape, split_op_param, absl::MakeSpan(start_indices),
          split_shape.dimensions()));

  // build the final dot
  std::vector<HloInstruction*> ops;
  if (split_is_lhs) {
    ops = {split_slice, join_op_param};
  } else {
    ops = {join_op_param, split_slice};
  }
  return builder_.AddInstruction(
      dot->CloneWithNewOperands(dot_shape, absl::MakeSpan(ops)));
}

StatusOr<HloInstruction*>
IntermediateTensorSplitterRewriteVisitor::Splitter::SplitLeafBroadcast(
    HloInstruction* broadcast, int64 split_dim, int64 split_size) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));

  // For a broadcast, we identify if we can split it by
  // changeing the broadcast itself, of if we have to
  // create slices of the underlying operand tensor.

  bool split_on_original_dim =
      absl::c_linear_search(broadcast->dimensions(), split_dim);

  int64 parameter_idx;
  Shape parameter_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                               operand->shape().dimensions());
  HloInstruction* new_operand;
  if (split_on_original_dim) {
    // we need to slice the parameter ...
    int64 operand_split_dim;
    for (int64 i = 0; i < broadcast->dimensions().size(); i++) {
      if (broadcast->dimensions(i) == split_dim) {
        operand_split_dim = i;
        break;
      }
    }

    parameter_shape.set_dimensions(operand_split_dim, split_size);

    std::vector<HloInstruction*> start_indices;
    for (int64 dim = 0; dim < operand->shape().dimensions_size(); dim++) {
      if (dim == operand_split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(builder_.AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(0))));
      }
    }

    HloInstruction* parameter;
    parameter_idx = AddParameter(operand, &parameter);

    new_operand = builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
        parameter_shape, parameter, absl::MakeSpan(start_indices),
        parameter_shape.dimensions()));
  } else {
    // This will be a parameter and we just modify the broadcast ...
    parameter_idx = AddParameter(operand, &new_operand);
  }

  Shape broadcast_shape = ShapeUtil::MakeShape(
      broadcast->shape().element_type(), broadcast->shape().dimensions());
  broadcast_shape.set_dimensions(split_dim, split_size);
  std::vector<HloInstruction*> params = {new_operand};
  return builder_.AddInstruction(
      broadcast->CloneWithNewOperands(broadcast_shape, absl::MakeSpan(params)));
}

StatusOr<HloInstruction*>
IntermediateTensorSplitterRewriteVisitor::Splitter::SplitLeafIota(
    HloInstruction* iota, int64 split_dim, int64 split_size) {
  CHECK(Match(iota, m::Iota()));

  // For an iota, we simply produce smaller iota and add the
  // loop offset to each parameter

  auto* iota_inst = DynCast<HloIotaInstruction>(iota);
  CHECK(iota_inst != nullptr);

  int64 parameter_idx = 0;
  Shape iota_shape = ShapeUtil::MakeShape(iota->shape().element_type(),
                                          iota->shape().dimensions());
  iota_shape.set_dimensions(split_dim, split_size);

  if (split_dim == iota_inst->iota_dimension()) {
    // The split is along the iota dimension, create offsets add
    // to a single internal iota
    HloInstruction* small_iota = builder_.AddInstruction(
        HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));

    HloInstruction* param;
    if (!ShapeUtil::SameElementType(offset_->shape(), small_iota->shape())) {
      Shape convert_shape = ShapeUtil::MakeShape(
          small_iota->shape().element_type(), offset_->shape().dimensions());
      param = builder_.AddInstruction(
          HloInstruction::CreateConvert(convert_shape, offset_));
    } else {
      param = offset_;
    }

    std::vector<int64> broadcast_dims = {};
    HloInstruction* broadcast =
        builder_.AddInstruction(HloInstruction::CreateBroadcast(
            iota_shape, param, absl::MakeSpan(broadcast_dims)));

    return builder_.AddInstruction(HloInstruction::CreateBinary(
        iota_shape, HloOpcode::kAdd, small_iota, broadcast));
  } else {
    // The split is not along an iota dimension, simply
    // create a smaller iota and add that as parameters.
    return builder_.AddInstruction(
        HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));
  }
}

HloInstruction*
IntermediateTensorSplitterRewriteVisitor::Splitter::BuildOutputTuple(
    int64 split_dim, int64 split_size, HloInstruction* original,
    HloInstruction* part, bool combine_with_sum) {
  // create the output init (broadcast off of 0)
  HloInstruction* output_init =
      original->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(original->shape().element_type())));
  std::vector<int64> broadcast_dims = {};
  output_init =
      original->parent()->AddInstruction(HloInstruction::CreateBroadcast(
          original->shape(), output_init, absl::MakeSpan(broadcast_dims)));
  HloInstruction* output;
  int64 output_idx = AddParameter(output_init, &output);

  HloInstruction* updated_output;
  if (combine_with_sum) {
    // we're splitting a dot on a dot dimension, this means
    // all that needs to be done is adding the part onto the
    // result (which is initialized as 0)
    updated_output = builder_.AddInstruction(HloInstruction::CreateBinary(
        output->shape(), HloOpcode::kAdd, output, part));
  } else {
    // slice part onto output
    std::vector<HloInstruction*> start_indices;
    for (int64 dim = 0; dim < output->shape().dimensions_size(); dim++) {
      if (dim == split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(builder_.AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(0))));
      }
    }
    updated_output =
        builder_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            output->shape(), output, part, start_indices));
  }

  // add split size to index
  HloInstruction* split_size_const = builder_.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(split_size)));
  HloInstruction* updated_index =
      builder_.AddInstruction(HloInstruction::CreateBinary(
          offset_->shape(), HloOpcode::kAdd, offset_, split_size_const));

  // collect idx, output and all parameters into a tuple ..
  std::vector<HloInstruction*> output_elements = {updated_index};
  for (int64 i = 1; i < param_->shape().tuple_shapes_size(); i++) {
    if (i != output_idx) {
      HloInstruction* get_tuple =
          builder_.AddInstruction(HloInstruction::CreateGetTupleElement(
              param_->shape().tuple_shapes(i), param_, i));
      output_elements.push_back(get_tuple);
    } else {
      output_elements.push_back(updated_output);
    }
  }
  return builder_.AddInstruction(HloInstruction::CreateTuple(output_elements));
}

Status IntermediateTensorSplitterRewriteVisitor::HandleDot(
    HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  // TODO: Handle the case where both operands can be
  //       split in a good way.

  bool can_split = false;
  bool split_is_lhs;
  std::vector<HloInstruction*> split_leafs;

  std::vector<int64> exclude_dims;
  std::vector<int64> orig_dims;
  for (int64 i = 0; i < lhs->shape().dimensions_size(); i++)
    orig_dims.push_back(i);

  if (OperandShouldBeSplit(lhs) &&
      OperandCanBeSplit(lhs, &split_leafs, &orig_dims, &exclude_dims)) {
    can_split = true;
    split_is_lhs = true;
  } else {
    exclude_dims.clear();
    orig_dims.clear();
    for (int64 i = 0; i < rhs->shape().dimensions_size(); i++)
      orig_dims.push_back(i);
    if (OperandShouldBeSplit(rhs) &&
        OperandCanBeSplit(rhs, &split_leafs, &orig_dims, &exclude_dims)) {
      can_split = true;
      split_is_lhs = false;
    }
  }

  if (can_split) {
    HloInstruction* split_inst = split_is_lhs ? lhs : rhs;
    int64 split_dim = BestSplitDim(split_inst, absl::MakeSpan(exclude_dims));
    if (split_dim == -1) {
      // Bail, we can't split this tensor into equally sized parts.
      return Status::OK();
    }

    bool combine_parts_with_sum =
        absl::c_linear_search(split_is_lhs ? dnums.lhs_contracting_dimensions()
                                           : dnums.rhs_contracting_dimensions(),
                              split_dim);

    int64 split_size = BestSplitSize(split_inst, split_dim);
    int64 split_count = split_inst->shape().dimensions(split_dim) / split_size;
    CHECK(split_size != -1);
    CHECK(split_count * split_size ==
          split_inst->shape().dimensions(split_dim));

    HloComputation::Builder body_builder(
        "intermediate_tensor_splitter_dot_body");
    Splitter splitter(body_builder, dot->parent(), absl::MakeSpan(split_leafs));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * comp_root,
        splitter.SplitInstruction(split_inst, split_dim, split_size));

    // Add final dot inside of the computation
    HloInstruction* reduce_param;
    int64 reduce_parameter_idx =
        splitter.AddParameter(split_is_lhs ? rhs : lhs, &reduce_param);

    Shape part_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                            dot->shape().dimensions());
    int64 dot_split_dim = split_dim;  // split dimension after dot occured
    if (split_is_lhs) {
      for (int64 c_dim : dnums.lhs_contracting_dimensions()) {
        if (c_dim < dot_split_dim) dot_split_dim--;
      }
    } else {
      for (int64 c_dim : dnums.rhs_contracting_dimensions()) {
        if (c_dim < dot_split_dim) dot_split_dim--;
      }
      dot_split_dim +=
          lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();
    }
    if (!combine_parts_with_sum)
      part_shape.set_dimensions(dot_split_dim, split_size);

    if (combine_parts_with_sum) {
      Shape sliced_shape =
          ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                               reduce_param->shape().dimensions());
      // FIXME: This assumes dots only contract once (which is currently always
      // true)
      int64 param_split_dim = split_is_lhs
                                  ? dnums.rhs_contracting_dimensions()[0]
                                  : dnums.lhs_contracting_dimensions()[0];
      sliced_shape.set_dimensions(param_split_dim, split_size);

      std::vector<HloInstruction*> start_indices;
      for (int64 dim = 0; dim < reduce_param->shape().dimensions_size();
           dim++) {
        if (dim == param_split_dim) {
          start_indices.push_back(splitter.offset());
        } else {
          start_indices.push_back(body_builder.AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(0))));
        }
      }
      reduce_param =
          body_builder.AddInstruction(HloInstruction::CreateDynamicSlice(
              sliced_shape, reduce_param, absl::MakeSpan(start_indices),
              sliced_shape.dimensions()));
    }

    std::vector<HloInstruction*> ops;
    if (split_is_lhs) {
      ops = {comp_root, reduce_param};
    } else {
      ops = {reduce_param, comp_root};
    }
    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(part_shape, absl::MakeSpan(ops)));

    HloInstruction* output_tuple = splitter.BuildOutputTuple(
        dot_split_dim, split_size, dot, part, combine_parts_with_sum);
    HloComputation* body =
        parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));

    // build the condition
    HloComputation::Builder cond_builder(
        "intermediate_tensor_splitter_dot_cond");
    HloInstruction* cond_param =
        cond_builder.AddInstruction(HloInstruction::CreateParameter(
            0, output_tuple->shape(), "loop_param"));
    HloInstruction* cond_offset =
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_tuple->shape().tuple_shapes(0), cond_param, 0));
    HloInstruction* offset_less_than =
        cond_builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int64>(split_size * split_count)));
    HloInstruction* compare =
        cond_builder.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
            ComparisonDirection::kLt));
    HloComputation* cond =
        parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

    // build the while and replace the original element with a get
    // tuple.
    int64 output_idx = output_tuple->shape().tuple_shapes().size() - 1;
    HloInstruction* init = dot->parent()->AddInstruction(
        HloInstruction::CreateTuple(splitter.parameters()));
    HloInstruction* loop = dot->parent()->AddInstruction(
        HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
    HloInstruction* result = dot->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(dot->shape(), loop, output_idx));

    return ReplaceInstruction(dot, result);
  }
}

Status IntermediateTensorSplitterRewriteVisitor::HandleReduce(
    HloInstruction* reduce) {
  if (!MatchSupportedReduce(reduce)) return Status::OK();

  // MatchSupportedReduce enforces that all inputs are of the
  // same shape, and that there is at least one operand!
  if (!OperandShouldBeSplit(reduce->mutable_operand(0))) return Status::OK();

  // TODO: This is a hack, I need to more seriously rethink the
  //       two pass system, to mark elements in a first pass and combine
  //       sections properly ...
  if (OperandShouldBeSplit(reduce)) return Status::OK();

  // If this is a multi-argument reduce, check if only one
  // result is used.
  if (reduce->shape().IsTuple() && reduce->user_count() > 1)
    return Status::OK();

  // MatchSupportedReduce enforces that all initializers are
  // scalars, so we only need to split the operands to the
  // reduce itself.
  int64 op_count = reduce->operand_count() / 2;
  std::vector<HloInstruction*> split_leafs;
  std::vector<int64> orig_dims;
  std::vector<int64> exclude_dims;
  for (int64 i = 0; i < op_count; i++) {
    orig_dims.clear();
    for (int64 i = 0; i < reduce->operand(i)->shape().dimensions_size(); i++)
      orig_dims.push_back(i);
    if (!OperandCanBeSplit(reduce->mutable_operand(i), &split_leafs, &orig_dims,
                           &exclude_dims))
      return Status::OK();
  }

  for (int64 reduce_dim : reduce->dimensions())
    exclude_dims.push_back(reduce_dim);

  int64 split_dim =
      BestSplitDim(reduce->mutable_operand(0), absl::MakeSpan(exclude_dims));
  if (split_dim == -1) {
    // Bail, we can't split this tensor into equally sized parts.
    return Status::OK();
  }

  int64 split_size = BestSplitSize(reduce->mutable_operand(0), split_dim);
  int64 split_count =
      reduce->mutable_operand(0)->shape().dimensions(split_dim) / split_size;
  CHECK(split_size != -1);
  CHECK(split_count * split_size ==
        reduce->mutable_operand(0)->shape().dimensions(split_dim));

  HloComputation::Builder body_builder(
      "intermediate_tensor_splitter_reduce_body");
  Splitter splitter(body_builder, reduce->parent(),
                    absl::MakeSpan(split_leafs));

  std::vector<HloInstruction*> operands;
  for (int64 i = 0; i < op_count; i++) {
    TF_ASSIGN_OR_RETURN(HloInstruction * split_op,
                        splitter.SplitInstruction(reduce->mutable_operand(i),
                                                  split_dim, split_size));
    operands.push_back(split_op);
  }

  // Add init parameters to computation
  for (int64 i = 0; i < op_count; i++) {
    HloInstruction* init_op;
    splitter.AddParameter(reduce->mutable_operand(i + op_count), &init_op);
    operands.push_back(init_op);
  }

  // Since initializers are scalars and operands are
  // not, this means the computation already supports
  // broadcasting (i.e. has only pointwise operands with
  // no set shape). We can just copy it directly!

  // TODO: I believe that this is true, but should double
  //       check...

  int64 reduce_split_dim = split_dim;  // split dim after reduce
  for (int64 r_dim : reduce->dimensions())
    if (r_dim < split_dim) reduce_split_dim--;

  Shape output_part_shape;
  HloInstruction *output_part, *old_output;
  if (reduce->shape().IsTuple()) {
    CHECK(reduce->user_count() == 1);
    old_output = reduce->users()[0];

    Shape new_reduce_shape = ShapeUtil::MakeTupleShape(
        absl::MakeSpan(reduce->shape().tuple_shapes()));
    for (int64 i = 0; i < new_reduce_shape.tuple_shapes_size(); i++) {
      new_reduce_shape.mutable_tuple_shapes(i)->set_dimensions(reduce_split_dim,
                                                               split_size);
    }
    HloInstruction* new_reduce = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(new_reduce_shape, operands));

    output_part_shape = ShapeUtil::MakeShape(old_output->shape().element_type(),
                                             old_output->shape().dimensions());
    output_part_shape.set_dimensions(reduce_split_dim, split_size);
    output_part =
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_part_shape, new_reduce, old_output->tuple_index()));
  } else {
    output_part_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                             reduce->shape().dimensions());
    output_part_shape.set_dimensions(reduce_split_dim, split_size);
    output_part = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(output_part_shape, operands));
    old_output = reduce;
  }

  HloInstruction* output_tuple = splitter.BuildOutputTuple(
      reduce_split_dim, split_size, old_output, output_part);
  HloComputation* body =
      parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));

  // build the condition
  HloComputation::Builder cond_builder(
      "intermediate_tensor_splitter_reduce_cond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, output_tuple->shape(), "loop_param"));
  HloInstruction* cond_offset =
      cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          output_tuple->shape().tuple_shapes(0), cond_param, 0));
  HloInstruction* offset_less_than =
      cond_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int64>(split_size * split_count)));
  HloInstruction* compare =
      cond_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
          ComparisonDirection::kLt));
  HloComputation* cond =
      parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

  // build the while and replace the original element with a get
  // tuple.
  int64 output_idx = output_tuple->shape().tuple_shapes().size() - 1;
  HloInstruction* init = reduce->parent()->AddInstruction(
      HloInstruction::CreateTuple(splitter.parameters()));
  HloInstruction* loop = reduce->parent()->AddInstruction(
      HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
  HloInstruction* result =
      reduce->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
          old_output->shape(), loop, output_idx));
  return ReplaceInstruction(old_output, result);
}

bool endsWith(string& str, string pattern) {
  if (pattern.size() > str.size()) return false;
  for (int i = 1; i <= pattern.size(); i++) {
    if (pattern[pattern.size() - i] != str[str.size() - i]) return false;
  }
  return true;
}

int64 IntermediateTensorSplitter::SplitTensorBytes() {
  string config = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  int64 raw = (int64)atoi(config.c_str());
  if (raw <= 0) return 134217728;  // 1 GiB

  if (endsWith(config, "GB") || endsWith(config, "gb"))
    return raw * 1000000000;  // 1e9
  else if (endsWith(config, "GiB"))
    return raw * 134217728;
  else if (endsWith(config, "MB") || endsWith(config, "mb"))
    return raw * 1000000;  // 1e6
  else if (endsWith(config, "MiB"))
    return raw * 1048576;
  else if (endsWith(config, "kB") || endsWith(config, "kb"))
    return raw * 1000;
  else if (endsWith(config, "kiB"))
    return raw * 1024;
  else
    return raw;  // interpret as bytes
}

StatusOr<bool> IntermediateTensorSplitter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  int64 split_size = SplitTensorBytes();
  IntermediateTensorSplitterRewriteVisitor rewrite(split_size, split_size,
                                                   module);
  return rewrite.RunOnModule(module);
}

}  // namespace xla
