// License TODO ....
#include "tensorflow/compiler/xla/service/intermediate_tensor_splitter.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class IntermediateTensorSplitterVisitor : public DfsHloRewriteVisitor {
  int64 max_intermediate_size;
  int64 target_intermediate_size;
  HloModule* parent_module;

 public:
  explicit IntermediateTensorSplitterVisitor(int64 max_intermediate_size,
                                             int64 target_intermediate_size,
                                             HloModule* parent_module)
      : max_intermediate_size(max_intermediate_size),
        target_intermediate_size(target_intermediate_size),
        parent_module(parent_module) {}

  // Determine if an operand is large enough such that we are
  // interested in splitting it.
  bool OperandShouldBeSplit(HloInstruction* inst);

  // Determine if an operand can be split by traversing it's
  // inputs until a splittable node is found.
  bool OperandCanBeSplit(HloInstruction* inst,
                         std::vector<HloInstruction*>* split_leafs = nullptr);

  // Matches any pointwise unary operator which has no side effects.
  static bool MatchPointwiseUnary(HloInstruction* inst,
                                  HloInstruction** operand = nullptr);

  // Matches any pointwise n-ary operator.
  static bool MatchPointwiseNary(
      HloInstruction* inst, std::vector<HloInstruction*>* operands = nullptr);

  // Matches a reduce operation where all operands have the same shape
  // and all initilizers are scalars.
  static bool MatchSupportedReduce(HloInstruction* inst);

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
    std::vector<std::vector<HloInstruction*>> parameters_;
    HloComputation::Builder& builder_;
    absl::Span<HloInstruction*> leafs_;

   public:
    explicit Splitter(HloComputation::Builder& builder,
                      absl::Span<HloInstruction*> leafs, int64 split_count)
        : builder_(builder), leafs_(leafs) {
      for (int64 i = 0; i < split_count; i++) {
        parameters_.push_back({});
      }
    }

    StatusOr<HloInstruction*> SplitInstruction(HloInstruction* inst,
                                               int64 split_dim,
                                               int64 split_size);

    StatusOr<HloInstruction*> SplitLeaf(HloInstruction* leaf, int64 split_dim,
                                        int64 split_size);

    int64 parameters_size() { return parameters_.size(); }

    std::vector<HloInstruction*>& parameters(int64 idx) {
      return parameters_.at(idx);
    }

    std::vector<std::vector<HloInstruction*>>& parameters() {
      return parameters_;
    }
  };
};

}  // namespace

bool IntermediateTensorSplitterVisitor::OperandShouldBeSplit(
    HloInstruction* inst) {
  return ShapeUtil::ElementsIn(inst->shape()) > max_intermediate_size;
}

bool IntermediateTensorSplitterVisitor::MatchPointwiseUnary(
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

bool IntermediateTensorSplitterVisitor::MatchPointwiseNary(
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

bool IntermediateTensorSplitterVisitor::MatchSupportedReduce(
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

bool IntermediateTensorSplitterVisitor::OperandCanBeSplit(
    HloInstruction* inst, std::vector<HloInstruction*>* split_leafs) {
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
    return OperandCanBeSplit(next, split_leafs);
  } else if (MatchPointwiseUnary(inst, &next)) {
    // This is a special case seperate from nary,
    // since we can make it tail recursive :)
    return OperandCanBeSplit(next, split_leafs);
  } else if (MatchPointwiseNary(inst, &next_vec)) {
    for (HloInstruction* next : next_vec) {
      // this path is not tail recursive :(
      if (!OperandCanBeSplit(next, split_leafs)) return false;
    }
    return true;
  } else {
    return false;
  }
}

int64 IntermediateTensorSplitterVisitor::BestSplitDim(
    HloInstruction* inst, absl::Span<const int64> excluded) {
  const Shape& shape = inst->shape();
  int64 best_dim = -1, best_size = 0;
  for (int64 i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) continue;
    if (shape.dimensions(i) > best_size && BestSplitSize(inst, i) != -1) {
      best_size = shape.dimensions(i);
      best_dim = i;
    }
  }
  return best_dim;
}

const int64 primes[64] = {2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,
                          37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,
                          83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137,
                          139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
                          197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                          263, 269, 271, 277, 281, 283, 293, 307, 311};

int64 IntermediateTensorSplitterVisitor::BestSplitSize(HloInstruction* inst,
                                                       int64 split_dim) {
  int64 split_size = inst->shape().dimensions(split_dim);
  int64 rest_size = ShapeUtil::ElementsIn(inst->shape()) / split_size;
  int64 factors[64];

  int64 tmp_size = split_size;
  for (int i = 0; i < 64; i++) {
    factors[i] = 0;
    while (tmp_size % primes[i] == 0) {
      factors[i]++;
      tmp_size /= primes[i];
    }
  }

  for (int i = 0; i < 64; i++)
    while (split_size * rest_size > target_intermediate_size &&
           factors[i]-- > 0)
      split_size /= primes[i];

  return split_size <= max_intermediate_size ? split_size : -1;
}

// TODO: This function want's to be split up ...
StatusOr<HloInstruction*>
IntermediateTensorSplitterVisitor::Splitter::SplitInstruction(
    HloInstruction* inst, int64 split_dim, int64 split_size) {
  HloInstruction *operand, *lhs, *rhs;
  std::vector<HloInstruction*> operands;

  if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
    // For the dot we identify the parameter to split and then
    // Generate the final dot operation, as well as the operand
    // vector.
    Shape dot_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                           inst->shape().dimensions());
    dot_shape.set_dimensions(split_dim, split_size);

    auto& dnums = inst->dot_dimension_numbers();
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

    // generate parameters for each split
    Shape split_shape = ShapeUtil::MakeShape(split_op->shape().element_type(),
                                             split_op->shape().dimensions());
    split_shape.set_dimensions(split_dim, split_size);

    std::vector<int64> start, limit, stride;
    for (int64 dim = 0; dim < split_op->shape().dimensions_size(); dim++) {
      start.push_back(0);
      limit.push_back(split_op->shape().dimensions(dim));
      stride.push_back(1);
    }

    int64 split_parameter_idx, join_parameter_idx;
    for (int64 i = 0, dims_done = 0; i < parameters_.size();
         i++, dims_done += split_size) {
      // build split parameter
      split_parameter_idx = parameters_.at(i).size();
      start[split_dim] = dims_done;
      limit[split_dim] = dims_done + split_size;
      HloInstruction* slice =
          split_op->parent()->AddInstruction(HloInstruction::CreateSlice(
              split_shape, split_op, absl::MakeSpan(start),
              absl::MakeSpan(limit), absl::MakeSpan(stride)));
      parameters_.at(i).push_back(slice);
      // attach join parameter
      join_parameter_idx = parameters_.at(i).size();
      parameters_.at(i).push_back(join_op);
    }

    // build the final dot
    HloInstruction* split_param =
        builder_.AddInstruction(HloInstruction::CreateParameter(
            split_parameter_idx, split_shape, "dot_split_tensor"));
    HloInstruction* join_param =
        builder_.AddInstruction(HloInstruction::CreateParameter(
            join_parameter_idx, join_op->shape(), "dot_join_tensor"));

    std::vector<HloInstruction*> ops;
    if (split_is_lhs) {
      ops = {split_param, join_param};
    } else {
      ops = {join_param, split_param};
    }
    return builder_.AddInstruction(
        inst->CloneWithNewOperands(dot_shape, absl::MakeSpan(ops)));
  } else if (Match(inst, m::Broadcast(m::Op(&operand)))) {
    // For a broadcast, we identify if we can split it by
    // changeing the broadcast itself, of if we have to
    // create slices of the underlying operand tensor.

    bool split_on_original_dim =
        absl::c_linear_search(inst->dimensions(), split_dim);

    int64 parameter_idx;
    Shape parameter_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), operand->shape().dimensions());
    if (split_on_original_dim) {
      int64 operand_split_dim;
      for (int64 i = 0; i < inst->dimensions().size(); i++) {
        if (inst->dimensions(i) == split_dim) {
          operand_split_dim = i;
          break;
        }
      }

      parameter_shape.set_dimensions(operand_split_dim, split_size);

      std::vector<int64> start, limit, stride;
      for (int64 dim = 0; dim < operand->shape().dimensions_size(); dim++) {
        start.push_back(0);
        limit.push_back(operand->shape().dimensions(dim));
        stride.push_back(1);
      }

      for (int64 i = 0, dims_done = 0; i < parameters_.size();
           i++, dims_done += split_size) {
        parameter_idx = parameters_.at(i).size();
        start[split_dim] = dims_done;
        limit[split_dim] = dims_done + split_size;
        HloInstruction* slice =
            operand->parent()->AddInstruction(HloInstruction::CreateSlice(
                parameter_shape, operand, absl::MakeSpan(start),
                absl::MakeSpan(limit), absl::MakeSpan(stride)));
        parameters_.at(i).push_back(slice);
      }
    } else {
      for (int64 i = 0; i < parameters_.size(); i++) {
        parameter_idx = parameters_.at(i).size();
        parameters_.at(i).push_back(operand);
      }
    }

    HloInstruction* parameter =
        builder_.AddInstruction(HloInstruction::CreateParameter(
            parameter_idx, parameter_shape, "broadcast"));

    Shape broadcast_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                                 inst->shape().dimensions());
    broadcast_shape.set_dimensions(split_dim, split_size);
    std::vector<HloInstruction*> params = {parameter};
    return builder_.AddInstruction(
        inst->CloneWithNewOperands(broadcast_shape, absl::MakeSpan(params)));
  } else if (Match(inst, m::Iota())) {
    // For an iota, we simply produce smaller iota and add a
    // constant offset to each parameter

    auto* iota_inst = DynCast<HloIotaInstruction>(inst);
    CHECK(iota_inst != nullptr);

    int64 parameter_idx = 0;
    Shape iota_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                            inst->shape().dimensions());
    iota_shape.set_dimensions(split_dim, split_size);

    if (split_dim == iota_inst->iota_dimension()) {
      // The split is along the iota dimension, create offsets and add
      // to a single internal iota
      HloInstruction* offset;
      for (int64 i = 0; i < parameters_.size(); i++) {
        parameter_idx = parameters_.at(i).size();
        offset = inst->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int64>(i * split_size)));
        parameters_.at(i).push_back(offset);
      }

      HloInstruction* iota = builder_.AddInstruction(
          HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));

      HloInstruction* param =
          builder_.AddInstruction(HloInstruction::CreateParameter(
              parameter_idx, offset->shape(), "iota_offset"));

      if (!ShapeUtil::SameElementType(param->shape(), iota->shape())) {
        Shape convert_shape = ShapeUtil::MakeShape(
            iota->shape().element_type(), offset->shape().dimensions());
        param = builder_.AddInstruction(
            HloInstruction::CreateConvert(convert_shape, param));
      }

      std::vector<int64> broadcast_dims = {};
      HloInstruction* broadcast =
          builder_.AddInstruction(HloInstruction::CreateBroadcast(
              iota_shape, param, absl::MakeSpan(broadcast_dims)));

      return builder_.AddInstruction(HloInstruction::CreateBinary(
          iota_shape, HloOpcode::kAdd, iota, broadcast));
    } else {
      // The split is not along an iota dimension, simply
      // create a smaller iota and add that as parameters.
      HloInstruction* iota = inst->parent()->AddInstruction(
          HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));
      for (std::vector<HloInstruction*>& params : parameters()) {
        parameter_idx = params.size();
        params.push_back(iota);
      }

      return builder_.AddInstruction(
          HloInstruction::CreateParameter(parameter_idx, iota_shape, "iota"));
    }
  } else if (Match(inst, m::Transpose(m::Op(&operand)))) {
    // For a transpose, the transpose might change which dimension is
    // being split. So we obtain the new split dimension and then
    // recursively a new operand to make a clone.
    int64 operand_split_dim = inst->dimensions(split_dim);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        SplitInstruction(operand, operand_split_dim, split_size));
    std::vector<HloInstruction*> ops = {new_operand};
    return builder_.AddInstruction(
        inst->CloneWithNewOperands(new_operand->shape(), absl::MakeSpan(ops)));
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
    CHECK(false);
  }
}

Status IntermediateTensorSplitterVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  // TODO: Handle the case where both operands can be
  //       split in a good way.

  bool can_split = false;
  bool split_is_lhs;
  std::vector<HloInstruction*> split_leafs;

  if (OperandShouldBeSplit(lhs) && OperandCanBeSplit(lhs, &split_leafs)) {
    can_split = true;
    split_is_lhs = true;
  } else if (OperandShouldBeSplit(rhs) &&
             OperandCanBeSplit(rhs, &split_leafs)) {
    can_split = true;
    split_is_lhs = false;
  }

  if (can_split) {
    HloInstruction* split_inst = split_is_lhs ? lhs : rhs;
    int64 split_dim = BestSplitDim(
        split_inst,
        absl::MakeSpan(split_is_lhs ? dnums.lhs_contracting_dimensions()
                                    : dnums.rhs_contracting_dimensions()));
    if (split_dim == -1) {
      // Bail, we can't split this tensor into equally sized parts.
      return Status::OK();
    }

    int64 split_size = BestSplitSize(split_inst, split_dim);
    int64 split_count = split_inst->shape().dimensions(split_dim) / split_size;
    CHECK(split_size != -1);
    CHECK(split_count * split_size ==
          split_inst->shape().dimensions(split_dim));

    HloComputation::Builder builder("intermediate_tensor_splitter_dot");
    Splitter splitter(builder, absl::MakeSpan(split_leafs), split_count);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * comp_root,
        splitter.SplitInstruction(split_inst, split_dim, split_size));

    // Add final dot inside of the computation
    int64 reduce_parameter_idx;
    for (int64 i = 0; i < splitter.parameters_size(); i++) {
      reduce_parameter_idx = splitter.parameters(i).size();
      splitter.parameters(i).push_back(split_is_lhs ? rhs : lhs);
    }
    HloInstruction* reduce_param =
        builder.AddInstruction(HloInstruction::CreateParameter(
            reduce_parameter_idx, split_is_lhs ? rhs->shape() : lhs->shape(),
            "reduce_tensor"));

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
    part_shape.set_dimensions(dot_split_dim, split_size);
    std::vector<HloInstruction*> ops;
    if (split_is_lhs) {
      ops = {comp_root, reduce_param};
    } else {
      ops = {reduce_param, comp_root};
    }
    HloInstruction* comp_dot = builder.AddInstruction(
        dot->CloneWithNewOperands(part_shape, absl::MakeSpan(ops)));

    HloComputation* comp =
        parent_module->AddEmbeddedComputation(builder.Build(comp_dot));

    std::vector<HloInstruction*> parts;
    for (auto operands : splitter.parameters()) {
      HloInstruction* call =
          dot->parent()->AddInstruction(HloInstruction::CreateCall(
              comp_dot->shape(), absl::MakeSpan(operands), comp));
      parts.push_back(call);
    }

    HloInstruction* concat =
        dot->parent()->AddInstruction(HloInstruction::CreateConcatenate(
            dot->shape(), absl::MakeSpan(parts), dot_split_dim));
    return ReplaceInstruction(dot, concat);
  }
}

Status IntermediateTensorSplitterVisitor::HandleReduce(HloInstruction* reduce) {
  if (!MatchSupportedReduce(reduce)) return Status::OK();

  // MatchSupportedReduce enforces that all inputs are of the
  // same shape, and that there is at least one operand!
  if (!OperandShouldBeSplit(reduce->mutable_operand(0))) return Status::OK();

  // MatchSupportedReduce enforces that all initializers are
  // scalars, so we only need to split the operands to the
  // reduce itself.
  int64 op_count = reduce->operand_count() / 2;
  std::vector<HloInstruction*> split_leafs;
  for (int64 i = 0; i < op_count; i++)
    if (!OperandCanBeSplit(reduce->mutable_operand(i), &split_leafs))
      return Status::OK();

  int64 split_dim = BestSplitDim(reduce->mutable_operand(0),
                                 absl::MakeSpan(reduce->dimensions()));
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

  HloComputation::Builder builder("intermediate_tensor_splitter_reduce");
  Splitter splitter(builder, absl::MakeSpan(split_leafs), split_count);

  std::vector<HloInstruction*> operands;
  for (int64 i = 0; i < op_count; i++) {
    TF_ASSIGN_OR_RETURN(HloInstruction * split_op,
                        splitter.SplitInstruction(reduce->mutable_operand(i),
                                                  split_dim, split_size));
    operands.push_back(split_op);
  }

  // Add init parameters to computation
  int64 parameter_start_idx;
  for (std::vector<HloInstruction*>& params : splitter.parameters()) {
    for (int64 i = 0; i < op_count; i++) {
      if (i == 0) parameter_start_idx = params.size();
      params.push_back(reduce->mutable_operand(i + op_count));
    }
  }
  for (int64 i = 0; i < op_count; i++) {
    const Shape& param_shape = reduce->operand(i + op_count)->shape();
    HloInstruction* init_op =
        builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_start_idx + i, param_shape, "init_op_param"));
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

  Shape new_reduce_shape;
  if (reduce->shape().IsTuple()) {
    new_reduce_shape = ShapeUtil::MakeTupleShape({});
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(reduce->shape()); i++) {
      const Shape& old_shape =
          ShapeUtil::GetTupleElementShape(reduce->shape(), i);
      Shape new_shape = ShapeUtil::MakeShape(old_shape.element_type(),
                                             old_shape.dimensions());
      new_shape.set_dimensions(reduce_split_dim, split_size);
      ShapeUtil::AppendShapeToTuple(new_shape, &new_reduce_shape);
    }
  } else {
    new_reduce_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                            reduce->shape().dimensions());
    new_reduce_shape.set_dimensions(reduce_split_dim, split_size);
  }

  HloInstruction* new_reduce = builder.AddInstruction(
      reduce->CloneWithNewOperands(new_reduce_shape, operands));
  HloComputation* comp =
      parent_module->AddEmbeddedComputation(builder.Build(new_reduce));

  std::vector<HloInstruction*> calls;
  for (auto operands : splitter.parameters()) {
    HloInstruction* call =
        reduce->parent()->AddInstruction(HloInstruction::CreateCall(
            new_reduce_shape, absl::MakeSpan(operands), comp));
    calls.push_back(call);
  }

  if (new_reduce_shape.IsTuple()) {
    // The output will be a tuple, we need to generate
    // concats for each output and update the
    // reduce shape for each output.
    std::vector<HloInstruction*> concats;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(new_reduce_shape); i++) {
      std::vector<HloInstruction*> items;
      for (HloInstruction* call : calls) {
        items.push_back(reduce->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                ShapeUtil::GetTupleElementShape(new_reduce_shape, i), call,
                i)));
      }
      concats.push_back(
          reduce->parent()->AddInstruction(HloInstruction::CreateConcatenate(
              ShapeUtil::GetTupleElementShape(reduce->shape(), i),
              absl::MakeSpan(items), reduce_split_dim)));
    }

    HloInstruction* tuple = reduce->parent()->AddInstruction(
        HloInstruction::CreateTuple(absl::MakeSpan(concats)));
    return ReplaceInstruction(reduce, tuple);
  } else {
    // Only have a single output, no tuple access is needed
    HloInstruction* concat =
        reduce->parent()->AddInstruction(HloInstruction::CreateConcatenate(
            reduce->shape(), absl::MakeSpan(calls), reduce_split_dim));
    return ReplaceInstruction(reduce, concat);
  }
}

StatusOr<bool> IntermediateTensorSplitter::Run(HloModule* module) {
  // TODO: Make the size limit configurable + find a better default
  int64 split_size = GetDebugOptionsFromFlags().xla_try_split_tensor_size();
  IntermediateTensorSplitterVisitor visitor(split_size, split_size, module);
  return visitor.RunOnModule(module);
}

}  // namespace xla
