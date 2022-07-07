/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_mco.h"

#include <set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
namespace xla {

namespace {

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;
void DebugPrint(std::string functionName, std::string content) {
  // std::cout << "[" << functionName << "]: " << content << std::endl;
  LOG(INFO) << "[" << functionName << "]: " << content;
}
void PrintCycle(const HloInstruction* child, DFSStack* dfs_stack) {
  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  while (!dfs.empty()) {
    bool found_next_instr = false;
    for (const auto& user : dfs.back()->users()) {
      if (user == child) {
        dfs.push_back(child);
        LOG(INFO) << "\n\nDirected cycle:\n  "
                  << absl::StrJoin(
                         dfs, "\n  ",
                         [](std::string* out, const HloInstruction* instr) {
                           out->append(instr->name());
                         });
        return;
      }
      if (!subgraph.contains(user) || visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      dfs.push_back(user);
      found_next_instr = true;
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }
}

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

// perform a pre-order DFS to find an existing chain
template <typename Visitor>
static Status DetectMatrixChainPreorderDFS(
    HloInstruction* root, Visitor* visitor,
    bool ignore_control_predecessors = false) {
  // Calculating the instruction count within a module can be expensive on large
  // models so only do it if the visit state is empty. This will help when the
  // same visitor is reused across many computations of a single module.
  if (visitor->VisitStateCapacity() == 0) {
    visitor->ReserveVisitStates(root->GetModule()->instruction_count());
  }

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);
  DebugPrint("DetectMatrixChainPreorderDFS",
             "Start, root = " + root->name() +
                 "opcode = " + HloOpcodeString(root->opcode()));

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0)
        << "[DetectMatrixChainPreorderDFS] " << current_id << ": "
        << current_node << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "[DetectMatrixChainPreorderDFS] "
              << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    dfs_stack.pop_back();
    bool is_matmul_node = MatrixChainDetector::CheckRealDot(current_node);
    VLOG(2) << "[DetectMatrixChainPreorderDFS] "
            << "Visiting HLO %" << current_node->name();
    DebugPrint("DetectMatrixChainPreorderDFS",
               "Visiting HLO = " + current_node->name() +
                   " opcode = " + HloOpcodeString(current_node->opcode()) +
                   " is_matmul = " + std::to_string(is_matmul_node));
    TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
    visitor->SetVisitState(current_id, Visitor::kVisited);
    if (!is_matmul_node) {
      // for ohter op, we just target the its child nodes in the current branch
      // as a single operand
      continue;
    }
    DebugPrint(
        "DetectMatrixChainPreorderDFS",
        "current_node = " + current_node->name() +
            " opcode = " + HloOpcodeString(current_node->opcode()) +
            " users.size = " + std::to_string(current_node->users().size()));

    const size_t old_dfs_stack_size = dfs_stack.size();
    // current_node is a dot op, must have 2 operands
    CHECK_EQ(current_node->operands().size(), 2);
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "DetectMatrixChainPreorderDFS A cycle is detected while visiting "
            "instruction %s",
            current_node->ToString());
      }
      DebugPrint("DetectMatrixChainPreorderDFS",
                 "current_node = " + current_node->name() +
                     " opcode = " + HloOpcodeString(current_node->opcode()) +
                     " Add Child " + child->name() +
                     " opcode = " + HloOpcodeString(child->opcode()));
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

// perform a pre-order DFS and rewrite nodes during traversal
// template <typename Visitor>
static Status PreorderDFSRewriter(HloInstruction* root,
                                  TransposeReduceSumUnfolder* visitor,
                                  bool& changed,
                                  bool ignore_control_predecessors = false) {
  // Calculating the instruction count within a module can be expensive on large
  // models so only do it if the visit state is empty. This will help when the
  // same visitor is reused across many computations of a single module.
  if (visitor->VisitStateCapacity() == 0) {
    visitor->ReserveVisitStates(root->GetModule()->instruction_count());
  }

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);
  // used to record roots of new copied chains
  DebugPrint("PreorderDFSRewriter",
             "Start, root = " + root->name() +
                 "opcode = " + HloOpcodeString(root->opcode()));

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0)
        << "[PreorderDFSRewriter] " << current_id << ": " << current_node
        << ": instruction may not have parent computation";
    typename TransposeReduceSumUnfolder::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == TransposeReduceSumUnfolder::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "[PreorderDFSRewriter] "
              << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    dfs_stack.pop_back();
    VLOG(2) << "[PreorderDFSRewriter] "
            << "Visiting HLO %" << current_node->name();
    DebugPrint("PreorderDFSRewriter",
               "Visiting HLO = " + current_node->name() +
                   " opcode = " + HloOpcodeString(current_node->opcode()));
    TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
    TF_RETURN_IF_ERROR(current_node->Visit(visitor));
    visitor->SetVisitState(current_id, TransposeReduceSumUnfolder::kVisited);
    TF_RETURN_IF_ERROR(visitor->Postprocess(current_node));
    if (visitor->OldNewMapContain(current_node)) {
      auto tmp = current_node;
      DebugPrint("PreorderDFSRewriter",
                 "Before replace current_node = " + current_node->name() +
                     " opcode = " + HloOpcodeString(current_node->opcode()) +
                     " OldNewMapContain() = " +
                     std::to_string(visitor->OldNewMapContain(tmp)));
      current_node = visitor->GetNewInst(current_node);
      visitor->DeleteOldInst(tmp);
      DebugPrint("PreorderDFSRewriter",
                 "Before replace current_node = " + current_node->name() +
                     " opcode = " + HloOpcodeString(current_node->opcode()) +
                     " OldNewMapContain() = " +
                     std::to_string(visitor->OldNewMapContain(tmp)));
      changed |= true;
    }
    DebugPrint(
        "PreorderDFSRewriter",
        "current_node = " + current_node->name() +
            " opcode = " + HloOpcodeString(current_node->opcode()) +
            " users.size = " + std::to_string(current_node->users().size()));

    const size_t old_dfs_stack_size = dfs_stack.size();
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "PreorderDFSRewriter A cycle is detected while "
            "visiting "
            "instruction %s",
            current_node->ToString());
      }
      DebugPrint("PreorderDFSRewriter",
                 "current_node = " + current_node->name() +
                     " opcode = " + HloOpcodeString(current_node->opcode()) +
                     " Add Child " + child->name() +
                     " opcode = " + HloOpcodeString(child->opcode()));
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

Status PreOrderAccept(HloComputation* computation,
                      TransposeReduceSumUnfolder* visitor, bool& changed) {
  return PreorderDFSRewriter(computation->root_instruction(), visitor, changed);
}

StatusOr<Literal> CreateOneConstVector(const PrimitiveType primitive_type,
                                       int64_t length) {
  switch (primitive_type) {
    case PrimitiveType::F16: {
      std::vector<half> raw_zero_vector(length, half(1));
      return LiteralUtil::CreateR1<half>(raw_zero_vector);
    }
    case PrimitiveType::BF16: {
      std::vector<bfloat16> raw_zero_vector(length, bfloat16(1));
      return LiteralUtil::CreateR1<bfloat16>(raw_zero_vector);
    }
    case PrimitiveType::F32: {
      std::vector<float> raw_zero_vector(length, float(1));
      return LiteralUtil::CreateR1<float>(raw_zero_vector);;
    }
    case PrimitiveType::F64: {
      std::vector<double> raw_zero_vector(length, double(1));
      return LiteralUtil::CreateR1<double>(raw_zero_vector);
    }
    case PrimitiveType::S8: {
      std::vector<int8> raw_zero_vector(length, int8(1));
      return LiteralUtil::CreateR1<int8>(raw_zero_vector);
    }
    case PrimitiveType::S16: {
      std::vector<int16> raw_zero_vector(length, int16(1));
      return LiteralUtil::CreateR1<int16>(raw_zero_vector);
    }
    case PrimitiveType::S32: {
      std::vector<int32> raw_zero_vector(length, int32(1));
      return LiteralUtil::CreateR1<int32>(raw_zero_vector);
    }
    case PrimitiveType::S64: {
      std::vector<int64_t> raw_zero_vector(length, int64_t(1));
      return LiteralUtil::CreateR1<int64_t>(raw_zero_vector);
    }
    case PrimitiveType::U8: {
      std::vector<uint8> raw_zero_vector(length, uint8(1));
      return LiteralUtil::CreateR1<uint8>(raw_zero_vector);
    }
    case PrimitiveType::U16: {
      std::vector<uint16> raw_zero_vector(length, uint16(1));
      return LiteralUtil::CreateR1<uint16>(raw_zero_vector);
    }
    case PrimitiveType::U32: {
      std::vector<uint32> raw_zero_vector(length, uint32(1));
      return LiteralUtil::CreateR1<uint32>(raw_zero_vector);
    }
    case PrimitiveType::U64: {
      std::vector<uint64> raw_zero_vector(length, uint64(1));
      return LiteralUtil::CreateR1<uint64>(raw_zero_vector);
    }
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", PrimitiveType_Name(primitive_type)));
  }
}

}  // namespace

bool TransposeReduceSumUnfolder::OldNewMapContain(HloInstruction* old_inst) {
  DebugPrint("TransposeReduceSumUnfolder::OldNewMapContain",
             "LookUp " + old_inst->name());
  return old_new_inst_map.contains(old_inst);
}
HloInstruction* TransposeReduceSumUnfolder::GetNewInst(
    HloInstruction* old_inst) {
  return old_new_inst_map[old_inst];
}

bool TransposeReduceSumUnfolder::DeleteOldInst(HloInstruction* old_inst) {
  return old_new_inst_map.erase(old_inst);
}

void TransposeReduceSumUnfolder::InsertOldNewMap(HloInstruction* old_inst,
                                                 HloInstruction* new_inst) {
  if (old_new_inst_map.contains(old_inst)) {
    DebugPrint("TransposeReduceSumUnfolder::old_new_inst_map",
               "Error: already contains: " + old_inst->name());
  }
  old_new_inst_map.insert({old_inst, new_inst});
}

bool TransposeReduceSumUnfolder::IsTransDot(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    return false;
  }
  const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      hlo->shape().dimensions_size() > 2 ||
      hlo->shape().dimensions_size() == 0) {
    // not m-m, m-v dot
    // donot need to handle v-v dot
    return false;
  }
  if (*(dnums.lhs_contracting_dimensions().begin()) == 1 &&
      *(dnums.rhs_contracting_dimensions().begin()) == 0) {
    // regular m-m or m-v dot
    return false;
  }

  if (*(dnums.lhs_contracting_dimensions().begin()) == 0 &&
      *(dnums.rhs_contracting_dimensions().begin()) == 1) {
    // m-m actual expression : N transdot M = N^T @ M^T
    return true;
  }
  if (*(dnums.lhs_contracting_dimensions().begin()) == 0 &&
      *(dnums.rhs_contracting_dimensions().begin()) == 0) {
    // m-v trans dot actual expression : v transdot M = M^T @ v
    return true;
  }
  if (*(dnums.lhs_contracting_dimensions().begin()) == 1 &&
      *(dnums.rhs_contracting_dimensions().begin()) == 1) {
    // m-m actual expression : M transdot N = M @ N^T
    return true;
  }
  return false;
}

bool TransposeReduceSumUnfolder::IsRegularDot(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    return false;
  }
  const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();

  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      hlo->shape().dimensions_size() > 2 ||
      hlo->shape().dimensions_size() == 0) {
    return false;
  }
  if (*(dnums.lhs_contracting_dimensions().begin()) == 1 &&
      *(dnums.rhs_contracting_dimensions().begin()) == 0) {
    // m-m or m-v dot
    return true;
  }
  return false;
}

bool TransposeReduceSumUnfolder::IsReduceSumDot(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kReduce) {
    return false;
  }
  HloInstruction* reduce_function = hlo->to_apply()->root_instruction();
  if (reduce_function->opcode() != HloOpcode::kAdd || !hlo->shape().IsArray() ||
      hlo->shape().dimensions_size() != 1) {
    // only consider reduce_sum(matrix) -> vector
    return false;
  }
  auto zero = LiteralUtil::CreateR0<float>(0);
  if (hlo->operand(1)->opcode() != HloOpcode::kConstant ||
      (hlo->operand(1)->literal() != zero)) {
    // is it possible Reduce Sum has non-zero init_value? if so, such
    // reduce sum cannot be target as matrix multiplication
    return false;
  }

  DebugPrint("TransposeReduceSumUnfolder::IsReduceSumDot",
             hlo->name() + " is reduce_sum");
  return true;
}

Status TransposeReduceSumUnfolder::HandleDot(HloInstruction* dot) {
  if (IsTransDot(dot)) {
    const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
    if (*(dnums.lhs_contracting_dimensions().begin()) == 0 &&
        *(dnums.rhs_contracting_dimensions().begin()) == 0) {
      // actual expression : v transdot M = M^T @ v
      StatusOr<HloInstruction*> status_or =
          MakeTransposeHlo(dot->mutable_operand(1), {1, 0});
      auto lhs_trans = std::move(status_or).ValueOrDie();
      status_or = MakeTransposeHlo(dot->mutable_operand(0), {0});
      auto rhs_trans = std::move(status_or).ValueOrDie();

      const Shape lhs_shape = lhs_trans->shape();
      DotDimensionNumbers dimension_numbers;
      dimension_numbers.add_lhs_contracting_dimensions(
          lhs_shape.dimensions_size() == 1 ? 0 : 1);
      dimension_numbers.add_rhs_contracting_dimensions(0);

      auto shape_status_or = ShapeInference::InferDotOpShape(
          lhs_trans->shape(), rhs_trans->shape(), dimension_numbers,
          dot->shape().element_type());
      Shape output_shape = std::move(shape_status_or).ValueOrDie();
      std::string temp_string =
          "IsTransDot, (M*v)^T InferDotOpShape:  operand1 = " +
          lhs_trans->name() + " shape = " + lhs_trans->shape().ToString() +
          " operand2 = " + rhs_trans->name() +
          " shape = " + rhs_trans->shape().ToString() +
          " inferred_output_shape = " + output_shape.ToString();
      DebugPrint("TransposeReduceSumUnfolder::HandleDot", temp_string);

      status_or =
          MakeDotHlo(lhs_trans, rhs_trans, dimension_numbers,
                     dot->precision_config(), dot->shape().element_type());
      HloInstruction* new_dot = std::move(status_or).ValueOrDie();
      InsertOldNewMap(dot, new_dot);
      return ReplaceInstruction(dot, new_dot);

    } else if (*(dnums.lhs_contracting_dimensions().begin()) == 1 &&
               *(dnums.rhs_contracting_dimensions().begin()) == 1) {
      // m-m actual expression : M transdot N = M @ N^T
      auto lhs = dot->mutable_operand(0);
      auto status_or = MakeTransposeHlo(dot->mutable_operand(1), {1, 0});
      auto rhs_trans = std::move(status_or).ValueOrDie();
      // cur_node = replace(cur_node, create_dot(trans_op0, trans_op1))

      const Shape lhs_shape = lhs->shape();
      DotDimensionNumbers dimension_numbers;
      dimension_numbers.add_lhs_contracting_dimensions(
          lhs_shape.dimensions_size() == 1 ? 0 : 1);
      dimension_numbers.add_rhs_contracting_dimensions(0);

      auto shape_status_or = ShapeInference::InferDotOpShape(
          lhs->shape(), rhs_trans->shape(), dimension_numbers,
          dot->shape().element_type());
      Shape output_shape = std::move(shape_status_or).ValueOrDie();
      std::string temp_string =
          "IsTransDot, (M*N)^T InferDotOpShape:  operand1 = " + lhs->name() +
          " shape = " + lhs->shape().ToString() +
          " operand2 = " + rhs_trans->name() +
          " shape = " + rhs_trans->shape().ToString() +
          " inferred_output_shape = " + output_shape.ToString();
      DebugPrint("TransposeReduceSumUnfolder::HandleDot", temp_string);

      status_or =
          MakeDotHlo(lhs, rhs_trans, dimension_numbers, dot->precision_config(),
                     dot->shape().element_type());
      HloInstruction* new_dot = std::move(status_or).ValueOrDie();
      InsertOldNewMap(dot, new_dot);
      return ReplaceInstruction(dot, new_dot);
    } else {
      // m-m actual expression : N transdot M = N^T @ M^T
      StatusOr<HloInstruction*> status_or =
          MakeTransposeHlo(dot->mutable_operand(0), {1, 0});
      auto lhs_trans = std::move(status_or).ValueOrDie();
      status_or = MakeTransposeHlo(dot->mutable_operand(1), {1, 0});
      auto rhs_trans = std::move(status_or).ValueOrDie();
      // cur_node = replace(cur_node, create_dot(trans_op0, trans_op1))

      const Shape lhs_shape = lhs_trans->shape();
      DotDimensionNumbers dimension_numbers;
      dimension_numbers.add_lhs_contracting_dimensions(
          lhs_shape.dimensions_size() == 1 ? 0 : 1);
      dimension_numbers.add_rhs_contracting_dimensions(0);

      auto shape_status_or = ShapeInference::InferDotOpShape(
          lhs_trans->shape(), rhs_trans->shape(), dimension_numbers,
          dot->shape().element_type());
      Shape output_shape = std::move(shape_status_or).ValueOrDie();
      std::string temp_string =
          "IsTransDot, (M*N)^T InferDotOpShape:  operand1 = " +
          lhs_trans->name() + " shape = " + lhs_trans->shape().ToString() +
          " operand2 = " + rhs_trans->name() +
          " shape = " + rhs_trans->shape().ToString() +
          " inferred_output_shape = " + output_shape.ToString();
      DebugPrint("TransposeReduceSumUnfolder::HandleDot", temp_string);

      status_or =
          MakeDotHlo(lhs_trans, rhs_trans, dimension_numbers,
                     dot->precision_config(), dot->shape().element_type());
      HloInstruction* new_dot = std::move(status_or).ValueOrDie();
      InsertOldNewMap(dot, new_dot);
      return ReplaceInstruction(dot, new_dot);
    }
  }
  return Status::OK();
}

Status TransposeReduceSumUnfolder::HandleTranspose(HloInstruction* transpose) {
  if (IsTransDot(transpose->operand(0))) {
    HloInstruction* lhs = transpose->mutable_operand(0)->mutable_operand(1);
    HloInstruction* rhs = transpose->mutable_operand(0)->mutable_operand(0);
    const Shape lhs_shape = lhs->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);

    auto shape_status_or = ShapeInference::InferDotOpShape(
        lhs->shape(), rhs->shape(), dimension_numbers,
        transpose->shape().element_type());
    Shape output_shape = std::move(shape_status_or).ValueOrDie();
    std::string temp_string =
        "Operand IsTransDot, InferDotOpShape:  operand1 = " + lhs->name() +
        " shape = " + lhs->shape().ToString() + " operand2 = " + rhs->name() +
        " shape = " + rhs->shape().ToString() +
        " inferred_output_shape = " + output_shape.ToString();
    DebugPrint("TransposeReduceSumUnfolder::HandleTranspose", temp_string);

    auto status_or = MakeDotHlo(lhs, rhs, dimension_numbers,
                                transpose->operand(0)->precision_config(),
                                transpose->shape().element_type());
    HloInstruction* new_dot = std::move(status_or).ValueOrDie();
    InsertOldNewMap(transpose, new_dot);
    return ReplaceInstruction(transpose, new_dot);

  } else if (IsRegularDot(transpose->operand(0))) {
    auto dnum_size = transpose->operand(0)->shape().dimensions_size();
    DebugPrint("TransposeReduceSumUnfolder::HandleTranspose",
               "IsRegularDot, dnum_size = " + std::to_string(dnum_size));
    if (dnum_size == 1) {
      // m-v dot, do not need to transpose
      return ReplaceInstruction(transpose, transpose->mutable_operand(0));
    } else {
      // m-m dot
      StatusOr<HloInstruction*> status_or = MakeTransposeHlo(
          transpose->mutable_operand(0)->mutable_operand(1), {1, 0});
      auto lhs_trans = std::move(status_or).ValueOrDie();
      status_or = MakeTransposeHlo(
          transpose->mutable_operand(0)->mutable_operand(0), {1, 0});
      auto rhs_trans = std::move(status_or).ValueOrDie();

      const Shape lhs_shape = lhs_trans->shape();
      DotDimensionNumbers dimension_numbers;
      dimension_numbers.add_lhs_contracting_dimensions(
          lhs_shape.dimensions_size() == 1 ? 0 : 1);
      dimension_numbers.add_rhs_contracting_dimensions(0);

      auto shape_status_or = ShapeInference::InferDotOpShape(
          lhs_trans->shape(), rhs_trans->shape(), dimension_numbers,
          transpose->shape().element_type());
      Shape output_shape = std::move(shape_status_or).ValueOrDie();
      std::string temp_string =
          "Operand IsRegularDot, InferDotOpShape:  operand1 = " +
          lhs_trans->name() + " shape = " + lhs_trans->shape().ToString() +
          " operand2 = " + rhs_trans->name() +
          " shape = " + rhs_trans->shape().ToString() +
          " inferred_output_shape = " + output_shape.ToString();
      DebugPrint("TransposeReduceSumUnfolder::HandleTranspose", temp_string);

      status_or = MakeDotHlo(lhs_trans, rhs_trans, dimension_numbers,
                             transpose->operand(0)->precision_config(),
                             transpose->shape().element_type());
      HloInstruction* new_dot = std::move(status_or).ValueOrDie();
      InsertOldNewMap(transpose, new_dot);
      return ReplaceInstruction(transpose, new_dot);
    }

  } else if (transpose->operand(0)->opcode() == HloOpcode::kTranspose) {
    std::string temp_string =
        "Operand IsTranspose, operand = : " +
        transpose->operand(0)->operand(0)->name() +
        " shape = " + transpose->operand(0)->operand(0)->shape().ToString();
    DebugPrint("TransposeReduceSumUnfolder::HandleTranspose", temp_string);
    InsertOldNewMap(transpose,
                    transpose->mutable_operand(0)->mutable_operand(0));
    return ReplaceInstruction(
        transpose, transpose->mutable_operand(0)->mutable_operand(0));
  }
  return Status::OK();
}

Status TransposeReduceSumUnfolder::HandleReduce(HloInstruction* reduce) {
  DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
             "Start " + reduce->ToString());
  if (!IsReduceSumDot(reduce)) {
    return Status::OK();
  }
  DebugPrint(
      "TransposeReduceSumUnfolder::HandleReduce",
      "Start " + reduce->name() + " dimension_size = " +
          std::to_string(reduce->shape().dimensions_size()) +
          " reduce_func = " +
          HloOpcodeString(reduce->to_apply()->root_instruction()->opcode()) +
          " operand.name = " + reduce->operand(0)->name() +
          " operand.dimension_size = " +
          std::to_string(reduce->operand(0)->shape().dimensions_size()) +
          " init_value.name = " + reduce->operand(1)->name());

  // replace reduce with dot
  HloInstruction* operand = reduce->mutable_operand(0);
  HloInstruction* init_value = reduce->mutable_operand(1);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  auto reduce_dims = reduce->shape().dimensions(0);
  DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
             "reduce_dims = " + std::to_string(reduce_dims));
  DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
             "operand->dimensions(0) = " +
                 std::to_string(operand->shape().dimensions(0)) +
                 " operand->dimensions(1) = " +
                 std::to_string(operand->shape().dimensions(1)));
  if (reduce_dims == operand->shape().dimensions(0)) {
    // M - v dot
    // constants
    DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
               "Is M-v dot, reduce_dims = " + std::to_string(reduce_dims));
    auto raw_zero_vector_status_or = CreateOneConstVector(
        reduce->shape().element_type(), operand->shape().dimensions(1));
    auto raw_zero_vector = std::move(raw_zero_vector_status_or).ValueOrDie();
    auto zero_vector_inst = reduce->parent()->AddInstruction(
        HloInstruction::CreateConstant(std::move(raw_zero_vector)));
    DebugPrint(
        "TransposeReduceSumUnfolder::HandleReduce",
        "Is M-v dot, zero_vector_inst = " + zero_vector_inst->ToString());
    auto lhs = reduce->mutable_operand(0);
    auto rhs = zero_vector_inst;
    // cur_node = replace(cur_node, create_dot(trans_op0, trans_op1))

    const Shape lhs_shape = lhs->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);

    auto shape_status_or = ShapeInference::InferDotOpShape(
        lhs->shape(), rhs->shape(), dimension_numbers,
        reduce->shape().element_type());
    Shape output_shape = std::move(shape_status_or).ValueOrDie();
    std::string temp_string =
        "M - v dot, InferDotOpShape:  operand1 = " + lhs->name() +
        " shape = " + lhs->shape().ToString() + " operand2 = " + rhs->name() +
        " shape = " + rhs->shape().ToString() +
        " inferred_output_shape = " + output_shape.ToString();
    DebugPrint("TransposeReduceSumUnfolder::HandleReduce", temp_string);

    auto status_or = MakeDotHlo(lhs, rhs, dimension_numbers, precision_config,
                                reduce->shape().element_type());
    HloInstruction* new_dot = std::move(status_or).ValueOrDie();
    InsertOldNewMap(reduce, new_dot);
    return ReplaceInstruction(reduce, new_dot);

  } else if (reduce_dims == operand->shape().dimensions(1)) {
    // v^T - M dot
    DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
               "Is v^T-M dot, reduce_dims = " + std::to_string(reduce_dims));
    auto raw_zero_vector_status_or = CreateOneConstVector(
        reduce->shape().element_type(), operand->shape().dimensions(0));
    auto raw_zero_vector = std::move(raw_zero_vector_status_or).ValueOrDie();
    auto zero_vector_inst = reduce->parent()->AddInstruction(
        HloInstruction::CreateConstant(std::move(raw_zero_vector)));
    DebugPrint(
        "TransposeReduceSumUnfolder::HandleReduce",
        "Is v^T-M dot, zero_vector_inst = " + zero_vector_inst->ToString());
    auto lhs = zero_vector_inst;
    auto rhs = reduce->mutable_operand(0);
    // cur_node = replace(cur_node, create_dot(trans_op0, trans_op1))

    const Shape lhs_shape = lhs->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);

    auto shape_status_or = ShapeInference::InferDotOpShape(
        lhs->shape(), rhs->shape(), dimension_numbers,
        reduce->shape().element_type());
    Shape output_shape = std::move(shape_status_or).ValueOrDie();
    std::string temp_string =
        "v^T - M dot, InferDotOpShape:  operand1 = " + lhs->name() +
        " shape = " + lhs->shape().ToString() + " operand2 = " + rhs->name() +
        " shape = " + rhs->shape().ToString() +
        " inferred_output_shape = " + output_shape.ToString();
    DebugPrint("TransposeReduceSumUnfolder::HandleReduce", temp_string);

    auto status_or = MakeDotHlo(lhs, rhs, dimension_numbers, precision_config,
                                reduce->shape().element_type());
    HloInstruction* new_dot = std::move(status_or).ValueOrDie();
    InsertOldNewMap(reduce, new_dot);
    return ReplaceInstruction(reduce, new_dot);
  }
  DebugPrint("TransposeReduceSumUnfolder::HandleReduce",
             "END " + reduce->ToString());
  return Status::OK();
}

Status ChainRecorder::Preprocess(HloInstruction* hlo) {
  if (!MatrixChainDetector::CheckRealDot(hlo)) {
    chain_map[chain_root].emplace_back(hlo);
    DebugPrint("ChainRecorder::Preprocess",
               "Add node: " + hlo->name() + " to root: " + chain_root->name() +
                   " chain_map[" + chain_root->name() +
                   "].size = " + std::to_string(chain_map[chain_root].size()));
  } else {
    DebugPrint("ChainRecorder::Preprocess",
               "Skip dot node: " + hlo->name() +
                   " opcode = " + HloOpcodeString(hlo->opcode()));
  }
  return Status::OK();
}
Status MatrixChainDetector::DetectMatrixChain(HloInstruction* chain_root) {
  std::deque<HloInstruction*> chain_roots{chain_root};
  while (!chain_roots.empty()) {
    HloInstruction* cur_root = chain_roots.front();
    chain_roots.pop_front();
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "Find a new chain_root:" + cur_root->name());
    ChainRecorder chain_recorder(cur_root);
    auto status = DetectMatrixChainPreorderDFS(cur_root, &chain_recorder);

    auto chain = chain_recorder.GetChain(cur_root);
    DebugPrint(
        "MatrixChainDetector::MatrixChainDetector",
        "Before insert chain_map.size() = " + std::to_string(chain_map.size()));
    chain_map.insert({cur_root, chain});
    DebugPrint(
        "MatrixChainDetector::MatrixChainDetector",
        "After insert chain_map.size() = " + std::to_string(chain_map.size()));
  }
  DebugPrint("MatrixChainDetector::MatrixChainDetector",
             "Finished chain_map.size() = " + std::to_string(chain_map.size()));
  return Status::OK();
}
bool MatrixChainDetector::CheckRealDot(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    return false;
  }

  const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      hlo->shape().dimensions_size() > 2 ||
      hlo->shape().dimensions_size() == 0) {
    // not m-m, m-v dot
    // do not need to handle v-v inner/outer product, since both of them won't
    // lead to better solution for inner-product, the result is a scalar for
    // outer-product, although the result is a matrix, but if we split the chain
    // from here and let two vector enter 2 sub-chain, the optimal solution is
    // the same
    return false;
  }

  if (*(dnums.lhs_contracting_dimensions().begin()) != 1 ||
      *(dnums.rhs_contracting_dimensions().begin()) != 0) {
    // since transpose op may be rewritten to dot op with
    // lhs_contracting_dimension = {0} rhs_contracting_dimension = {1}
    // such dot is actuall a tranpose and is not associative with other dots.
    DebugPrint(
        "MatrixChainDetector::CheckRealDot",
        " kDot check fail: inst.name:" + hlo->name() + " opcode = " +
            HloOpcodeString(hlo->opcode()) + " dimensions_size() = " +
            std::to_string(hlo->shape().dimensions_size()) +
            " lhs_contracting_dimension = " +
            std::to_string(*(dnums.lhs_contracting_dimensions().begin())) +
            " rhs_contracting_dimension = " +
            std::to_string(*(dnums.rhs_contracting_dimensions().begin())));

    return false;
  }
  DebugPrint("MatrixChainDetector::CheckRealDot",
             "End check dot op: " + hlo->name() + " True");
  return true;
}
Status MatrixChainDetector::Preprocess(HloInstruction* hlo) {
  DebugPrint("MatrixChainDetector::Preprocess", "Start " + hlo->ToString());
  // skip 2D dot op but if it is the root_instruction, then it must be the root
  // of a matrix chain

  if (CheckRealDot(hlo)) {
    if (hlo != hlo->parent()->root_instruction()) {
      return Status::OK();
    } else {
      DebugPrint("MatrixChainDetector::Preprocess",
                 "root_instruction is dot, inst.name:" + hlo->name() +
                     " opcode = " + HloOpcodeString(hlo->opcode()));
      TF_RETURN_IF_ERROR(DetectMatrixChain(hlo));
      DebugPrint(
          "MatrixChainDetector::Preprocess",
          "After DetectMatrixChain(" + hlo->name() +
              ") chain_map.size() = " + std::to_string(chain_map.size()));
      return Status::OK();
    }
  }

  for (auto op : hlo->operands()) {
    DebugPrint("MatrixChainDetector::Preprocess",
               "operand.name:" + op->name() +
                   " opcode = " + HloOpcodeString(op->opcode()));
    if (!CheckRealDot(op)) {
      // Only consider 2D dot product for now
      VLOG(10) << "[MatrixChainDetector]: Can only optimize 2D, non-batch dot "
                  "operations.";
      DebugPrint("MatrixChainDetector::Preprocess",
                 "Can only optimize 2D non-batch dot operations.");
      continue;
    }
    // current node != kDot, child op = kDot, child op is the root of a matrix
    // chain
    if (chain_map.contains(op)) {
      // find a reused chain which has already been added into chain_map
      continue;
    }
    TF_RETURN_IF_ERROR(DetectMatrixChain(op));
    DebugPrint("MatrixChainDetector::Preprocess",
               "After DetectMatrixChain(" + op->name() +
                   ") chain_map.size() = " + std::to_string(chain_map.size()));
  }

  return Status::OK();
}

StatusOr<HloInstruction*> HloMCO::CopyResuableSubgraph(HloInstruction* inst) {
  // deprecated
  std::vector<HloInstruction*> users;
  users.assign(inst->users().begin() + 1, inst->users().end());
  HloInstruction* new_inst = inst->parent()->AddInstruction(inst->Clone());
  inst->ReplaceUsesWith(users, new_inst);
  return new_inst;
}

StatusOr<HloInstruction*> HloMCO::ConstructOptimalChain(
    HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
    std::vector<HloInstruction*>& chain_instructions) {
  DebugPrint("HloMCO::ConstructOptimalChain",
             "Start, root = " + orig_root->name());
  HloInstruction* optimal_root = nullptr;
  std::vector<HloInstruction*> subgraph_stack;
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, 0, chain_instructions.size() - 1,
      subgraph_stack));
  DebugPrint("HloMCO::ConstructOptimalChain",
             "subgraph_stack.size = " + std::to_string(subgraph_stack.size()));
  CHECK_EQ(subgraph_stack.size(), 1);
  optimal_root = subgraph_stack.back();
  return optimal_root;
}

Status HloMCO::ConstructOptimalChainHelper(
    HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
    std::vector<HloInstruction*>& chain_instructions, int64_t start_index,
    int64_t end_index, std::vector<HloInstruction*>& subgraph_stack) {
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "Start, root = " + orig_root->name() + " (start,end) = (" +
          std::to_string(start_index) + "," + std::to_string(end_index) + ") " +
          "subgraph_stack.size = " + std::to_string(subgraph_stack.size()));
  auto create_dot = [&](HloInstruction* l, HloInstruction* r) {
    std::string temp_string = "Start:  operand1 = " + l->name() +
                              " shape = " + l->shape().ToString() +
                              " operand2 = " + r->name() +
                              " shape = " + r->shape().ToString();
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot", temp_string);
    const Shape lhs_shape = l->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    auto status_or = ShapeInference::InferDotOpShape(
        l->shape(), r->shape(), dimension_numbers, l->shape().element_type());
    Shape output_shape = std::move(status_or).ValueOrDie();

    temp_string = "InferDotOpShape:  operand1 = " + l->name() +
                  " shape = " + l->shape().ToString() +
                  " operand2 = " + r->name() +
                  " shape = " + r->shape().ToString() +
                  " inferred_output_shape = " + output_shape.ToString();
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot", temp_string);

    // for newly created instruction, we need to save it to the computation
    HloInstruction* new_matmul_inst_ptr = l->parent()->AddInstruction(
        HloInstruction::CreateDot(output_shape, l, r, dimension_numbers,
                                  orig_root->precision_config()));
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot",
               "create new matmul: " + new_matmul_inst_ptr->name() +
                   "shape = " + new_matmul_inst_ptr->shape().ToString() +
                   " operand1 = " + l->name() + " operand2 = " + r->name());
    subgraph_stack.emplace_back(new_matmul_inst_ptr);
    DebugPrint("HloMCO::ConstructOptimalChainHelper::create_dot",
               "After insert subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name() +
                   " shape = " + subgraph_stack.back()->shape().ToString());
  };

  if (start_index == end_index) {
    DebugPrint(
        "HloMCO::ConstructOptimalChainHelper",
        "Add single operand: " + chain_instructions[start_index]->name());
    // for single operand, it has already been stored in the compoutation
    subgraph_stack.emplace_back(chain_instructions[start_index]);
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "After insert single operand subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name());
    return Status::OK();
  }

  if (start_index == end_index - 1) {
    // construction a new matmul op
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "Add single matmul: " + chain_instructions[start_index]->name() +
                   " * " + chain_instructions[end_index]->name());
    create_dot(chain_instructions[start_index], chain_instructions[end_index]);
    DebugPrint("HloMCO::ConstructOptimalChainHelper",
               "After insert single matmul subgraph_stack.size= " +
                   std::to_string(subgraph_stack.size()) +
                   " top = " + subgraph_stack.back()->name());
    return Status::OK();
  }
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "First interval = [" + std::to_string(start_index) + "," +
                 std::to_string(solution[start_index][end_index]) + "] " +
                 "Second interval = [" +
                 std::to_string(solution[start_index][end_index] + 1) + "," +
                 std::to_string(end_index) + "]");
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, start_index,
      solution[start_index][end_index], subgraph_stack));
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions,
      solution[start_index][end_index] + 1, end_index, subgraph_stack));

  // since this is a stack, the right_operand is on the top of left_operand
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "Before combile subgraph_stack.size= " +
                 std::to_string(subgraph_stack.size()) +
                 " top = " + subgraph_stack.back()->name());
  HloInstruction* right_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "combile right_operand =" + right_operand->name() +
          " subgraph_stack.size= " + std::to_string(subgraph_stack.size()));
  HloInstruction* left_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "combile left_operand =" + left_operand->name() +
          " subgraph_stack.size= " + std::to_string(subgraph_stack.size()));
  create_dot(left_operand, right_operand);
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "After combile subgraph_stack.size= " +
                 std::to_string(subgraph_stack.size()) +
                 " top = " + subgraph_stack.back()->name());

  return Status::OK();
}

StatusOr<HloInstruction*> HloMCO::ComputeOptimalChainOrder(
    HloInstruction* root, std::vector<HloInstruction*>& chain) {
  DebugPrint("HloMCO::ComputeOptimalChainOrder",
             "chain_root = " + root->name() +
                 " chain_length = " + std::to_string(chain.size()));
  HloInstruction* optimal_root = nullptr;
  int64_t chain_length = chain.size();
  // sizes[i] stores the number of rows of operand[i]
  // sizes[i+1] stores the number of columns of operand[i]
  std::vector<int64_t> sizes(chain_length + 1, 0);
  for (auto i = 0; i < chain_length; ++i) {
    CHECK_LE(chain[i]->shape().rank(), 2);
    if (chain[i]->shape().rank() == 1) {
      // vector operand
      // a vector in XLA is row vector or column vector? For now consider
      // it as column vector
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = 1;
    } else if (chain[i]->shape().rank() == 2) {
      // matrix operand
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = chain[i]->shape().dimensions(1);
    }
  }
  // solution[i][j] stores optimal break point in
  // subexpression from i to j.
  std::vector<std::vector<int64_t>> solution(
      chain_length, std::vector<int64_t>(chain_length, 0));
  /* costs[i,j] = Minimum number of scalar multiplications
        needed to compute the matrix A[i]A[i+1]...A[j] =
        A[i..j] */
  std::vector<std::vector<int64_t>> costs(
      chain_length,
      std::vector<int64_t>(chain_length, std::numeric_limits<int64_t>::max()));
  // cost is zero when multiplying one matrix.
  for (int64_t i = 0; i < chain_length; i++) costs[i][i] = 0;

  // L is chain length.
  // Dynamic Programming to find the optimal computing order
  for (int64_t L = 2; L <= chain_length; L++) {
    for (int64_t i = 0; i <= chain_length - L; i++) {
      // L = 2:             L = n:
      // i = 0 -> n-2       i = 0 -> 0
      // j = 1 -> n-1       j = n-1 -> n-1
      int64_t j = i + L - 1;
      // [i,j] is the [start,end] index of the current subchain
      for (int64_t k = i; k <= j - 1; k++) {
        // compute
        int64_t cost = costs[i][k] + costs[k + 1][j] +
                       sizes[i] * sizes[k + 1] * sizes[j + 1];
        if (cost < costs[i][j]) {
          costs[i][j] = cost;
          // Each entry solution[i,j]=k shows
          // where to split the product arr
          // i,i+1....j to [i,k] * [k+1,j] for the minimum cost.
          solution[i][j] = k;
        }
      }
    }
  }
  auto status_or = ConstructOptimalChain(root, solution, chain);
  optimal_root = std::move(status_or).ValueOrDie();
  return optimal_root;
}

StatusOr<bool> HloMCO::ChainOptimize(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        chain_map) {
  DebugPrint("HloMCO::ChainOptimize", "Start");
  bool changed = false;
  for (auto& item : chain_map) {
    DebugPrint("HloMCO::ChainOptimize",
               "optimize chain_root = " + item.first->name());
    auto status_or = ComputeOptimalChainOrder(item.first, item.second);
    HloInstruction* new_instruction = std::move(status_or).ValueOrDie();
    DebugPrint(
        "HloMCO::ChainOptimize",
        "Finish optimization, new_chain_root = " + new_instruction->name());
    DebugPrint("HloMCO::ChainOptimize",
               "Before replace, chain_root.user.size = " +
                   std::to_string(item.first->users().size()));

    item.first->ReplaceAllUsesWith(new_instruction);
    if (new_instruction == item.first->parent()->root_instruction()) {
      DebugPrint("HloMCO::ChainOptimize",
                 "Replace computation root success, new_root: " +
                     new_instruction->name());
    }
    DebugPrint("HloMCO::ChainOptimize",
               "After replace, chain_root.user.size = " +
                   std::to_string(item.first->users().size()));

    changed = true;
  }
  return changed;
}

StatusOr<bool> HloMCO::Run(HloModule* module) {
  bool changed = false;
  TF_RET_CHECK(!module->name().empty());

  if (module->entry_computation()->IsFusionComputation()) {
    return InvalidArgument(
        "Module entry computation cannot be a fusion computation");
  }
  DebugPrint("HloMCO::Run", "Start Run");
  for (auto* computation : module->MakeNonfusionComputations()) {
    DebugPrint("HloMCO::Run", "computation: " + computation->ToString());
    // DebugPrint("HloMCO::Run", "start cleanup_visitor");
    // CleanupVisitor cleanup_visitor;
    // TF_RETURN_IF_ERROR(computation->Accept(&cleanup_visitor));

    TransposeReduceSumUnfolder transpose_unfolder;
    DebugPrint("HloMCO::Run", "start transpose_unfolder");
    PreOrderAccept(computation, &transpose_unfolder, changed);
    DebugPrint("HloMCO::Run", "start matriox_chain_detector");
    MatrixChainDetector matrix_chain_detector;
    // detection matrix chain on the whithin the computation;
    TF_RETURN_IF_ERROR(computation->Accept(&matrix_chain_detector));
    DebugPrint("HloMCO::Run", "finish matriox_chain_detector");
    auto chain_map = matrix_chain_detector.GetChainMap();
    DebugPrint("HloMCO::Run",
               "chain_map.size = " + std::to_string(chain_map.size()));
    TF_ASSIGN_OR_RETURN(bool changed_for_computation,
                        ChainOptimize(computation, chain_map));
    changed |= changed_for_computation;
    computation->Cleanup();
    DebugPrint("HloMCO::Run",
               "After optimization computation: " + computation->ToString());
  }
  // After these pass, also need to run HloDCE and HloCSE
  return changed;
}

}  // namespace xla
