

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_

#include <stack>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"
namespace xla {

//  This pass tries to detect and optimize matrix multiplication chains in
//  a graph. For example, a matrix chain ABC whose dimensions are
//  [100,10],[10,20] and [20,10] respectively. The expression would be computed
//  in the order of (AB)C, however, if we could preform the computation in the
//  order of A(BC), less memory and faster speed could be achieved
class HloMCO : public HloModulePass {
 public:
  explicit HloMCO(bool only_fusion_computations = false)
      : only_fusion_computations_(only_fusion_computations) {}
  ~HloMCO() override = default;
  absl::string_view name() const override { return "mco"; }

  //  Run MCO on the given module.
  //  Input:
  //    module: the target module to run the optimization
  //  Output:
  //    a bool value which indicates whether the module was changed (matrix
  //    chains were found and optimizesd).
  StatusOr<bool> Run(HloModule* module) override;

  //  Optimize matrix chains in a given computation
  //  Input:
  //    computation: the target computation
  //    chain_map: a hashtable where the key is the root node of a matrix chain
  //    and the value is
  //      all operands involved in the chain.
  //    reduce_one_vector_to_orig_init_val: a hashtable where the key is the
  //    constructed all-ones vector and the value is
  //      the initial value operand of the original reduce_sum instruction.
  //  Output:
  //    a bool value which indicates whether the computation was changed (matrix
  //    chains were found and optimized)
  StatusOr<bool> ChainOptimize(
      HloComputation* computation,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
          chain_map,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);

  //  Compute the optimal order and construct an optimal chain to replace
  //  the given matrix chain.
  //  Input:
  //    root: the root node of a matrix chain
  //    chain: all operands involved in the matrix chain
  //    reduce_one_vector_to_orig_init_val: a hashtable where the key is the
  //    constructed all-ones vector and the value is
  //      the initial value operand of the original reduce_sum instruction.
  //  Output:
  //    the instruction pointer to the new optimal matrix chain
  StatusOr<HloInstruction*> ComputeOptimalChainOrder(
      HloInstruction* root, std::vector<HloInstruction*>& chain,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);

  //  Given a matrix chain and its optimal computational order, construct
  //  the optimal chain.
  //  Input:
  //    orig_root: the root node of a matrix chain
  //    solution: the solution result from dynamic programming MCO algorithm
  //        which represents the optimal order of the chain with orig_root as
  //        its root
  //    chain_instructions: all operands involved in the matrix chain
  //    reduce_one_vector_to_orig_init_val: a hashtable where the key is the
  //      constructed all-ones vector and the value is the initial value operand
  //      of the original reduce_sum instruction.
  //  Output:
  //    the instruction pointer to the new optimal matrix chain
  StatusOr<HloInstruction*> ConstructOptimalChain(
      HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
      std::vector<HloInstruction*>& chain_instructions,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);

  //  A helper function to construct a new optimal chain recursively
  //  Input:
  //    orig_root: the root node of a matrix chain
  //    solution: the solution result from dynamic programming MCO algorithm
  //        which represents the optimal order of the chain with orig_root as
  //        its root
  //    chain_instructions: all operands involved in the matrix chain
  //    start_index: the start index of the subchain in chain_instructions
  //    end_index: the end index of the subchain in chain_instructions
  //    subgraph_stack: a helper stack to construct the whole chain, the caller
  //      of this method should use an empty stack as the parameter
  //    reduce_one_vector_to_orig_init_val: a hashtable where the key is the
  //      constructed all-ones vector and the value is the initial value operand
  //      of the original reduce_sum instruction.
  //  Output:
  //    a status which indicates whether the constructing process is successful
  //    and the result instruction would be the top element of subgraph_stack.
  Status ConstructOptimalChainHelper(
      HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
      std::vector<HloInstruction*>& chain_instructions, int64_t start_index,
      int64_t end_index, std::vector<HloInstruction*>& subgraph_stack,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);
  const PrecisionConfig& precision_config() const { return precision_config_; }
  Status SetPrecisionConfig(PrecisionConfig& precision_config) {
    precision_config_ = precision_config;
  }

 private:
  const bool only_fusion_computations_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

// A helper visitor class used to record a matrix multiplication chain
class ChainRecorder : public DfsHloVisitorWithDefault {
 public:
  //  The constructor fuction
  //  Input:
  //    input_chain_root: the root of a matrix chain to be recorded
  ChainRecorder(HloInstruction* input_chain_root)
      : chain_root(input_chain_root) {}
  Status DefaultAction(HloInstruction* hlo) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction* hlo) override { return Status::OK(); }
  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>

  //  A helper fuction to get the recorded matrix chain map
  //  Output:
  //    chain_map: the recorded matrix chain map
  GetChainMap() {
    return chain_map;
  }

  //  A helper fuction to get the length of the recorded matrix cahin
  //  Output:
  //    the length of the recorded matrix cahin
  int64_t GetChainLength(HloInstruction* root) {
    return chain_map[root].size();
  }

  //  A helper fuction to get operands involved in the recorded matrix chain
  //  Output:
  //    a vector contains all operands in the recorded matrix chain
  std::vector<HloInstruction*> GetChain(HloInstruction* root) {
    return chain_map[root];
  }
  //  A helper fuction to remove the chain from the recorded matrix chain map
  //  Output:
  //    A status indicates whether the deletion is successful
  Status RemoveChain(HloInstruction* root) {
    chain_map.erase(root);
    return Status::OK();
  }

 private:
  // the recorded matrix chain in the form of (chain_root_ptr,
  // vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
  // the root of a matrix chain to be recorded
  HloInstruction* chain_root;
};

// A visitor class used to detect and record all matrix multiplication chains
// in a graph, which is used in the first phase in the whole MCO process
class MatrixChainDetector : public DfsHloVisitorWithDefault {
 public:
  MatrixChainDetector() {}

  //  A helper fuction to get the recorded matrix chain map
  //  Output:
  //    chain_map: the recorded matrix chain map
  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  Status DefaultAction(HloInstruction* hlo) override { return Status::OK(); }
  Status Preprocess(HloInstruction* hlo) override;

  //  A helper fuction to check if a dot instruction is a m-m, m-v dot operation
  //  Output:
  //    a bool value indicates if a dot instruction is a m-m, m-v dot operation
  static bool CheckRealDot(HloInstruction* hlo);

  Status FinishVisit(HloInstruction* hlo) override { return Status::OK(); }

  //  Record a matrix multiplication chain
  //  Input:
  //    chain_root: the root of a matrix chain to be recorded
  //  Output:
  //    a status indicates if the recording process is successful
  Status DetectMatrixChain(HloInstruction* chain_root);

 private:
  // the recorded matrix chain in the form of (chain_root_ptr,
  // vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
};

// A  post-order visitor which convert einsum to regular matrix/vector product
//  convert reduce_sum to regular m-v dot and unfold transpose
class EinSumReduceSumConverter : public DfsHloRewriteVisitor {
 public:
  //  The handler of dot operation to convert einsum to regular matrix/vector
  //  product and unfold its transpose operand Input:
  //    dot: the currently visited dot operation
  //  Output:
  //    a status indicates whether the processing is successful
  Status HandleDot(HloInstruction* dot) override;

  //  The handler of reduce operation to convert reduce_sum to regular
  //  matrix-vector product and unfold its transpose operand Input:
  //    reduce: the currently visited reduce operation
  //  Output:
  //    a status indicates whether the processing is successful
  Status HandleReduce(HloInstruction* reduce) override;

  //  A helper function used to unfold transpose operands in a matrix
  //  multiplication chain Input:
  //    trans_stack: A stack contains [left_operand,right_operand] of the root
  //    dot instruction of
  //      a matrix chain
  //  Output:
  //    a status indicates whether the processing is successful
  Status TransposeSinker(std::stack<HloInstruction*> trans_stack);

  //  A helper fuction to check if a reduce instruction is a reduce_sum operation
  //  Output:
  //    a bool value indicates if a reduce instruction is a reduce_sum operation
  bool IsReduceSumDot(const HloInstruction* hlo);

  // A helper function to check if the dimension order of a instruction matches
  // the order of its dimension indices
  //  Input:
  //    reduce: the currently visited reduce operation
  //  Output:
  //    a bool value indicates if the dimension order of a instruction matches
  //    the order of its dimension indices
  bool DimsAreIndexes(HloInstruction* inst) {
    for (auto i = 0; i < inst->dimensions().size(); i++) {
      if (i != inst->dimensions(i)) {
        return false;
      }
    }
    return true;
  }

  // A helper function to check if a transpose is trivial, i.e. it doesn't
  // change the dimension order
  //  Input:
  //    transpose: the target transpose operation
  //  Output:
  //    a bool value indicates if the transpose is trivial
  bool IsTrivialTranspose(HloInstruction* transpose) {
    return DimsAreIndexes(transpose);
  }
  absl::flat_hash_map<HloInstruction*, HloInstruction*>&
  GetReduceOneVetorMap() {
    return reduce_one_vector_to_orig_init_val;
  }

 private:
  //  a hashtable where the key is the constructed all-ones vector and the value
  //  is the initial value operand of the original reduce_sum instruction.
  absl::flat_hash_map<HloInstruction*, HloInstruction*>
      reduce_one_vector_to_orig_init_val;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
