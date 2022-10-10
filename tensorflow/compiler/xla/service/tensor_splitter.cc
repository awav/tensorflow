// License TODO ....
#include "tensorflow/compiler/xla/service/tensor_splitter.h"

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

// SplitProperties is a tuple with (size, count, rest) elements
using SplitProperties = std::tuple<int64_t, int64_t, int64_t>;

#define CONSTANT_INT64(number) \
  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(number))

class TensorSplitterRewriteVisitor : public DfsHloRewriteVisitor {
  int64_t max_size_threshold;
  int64_t target_split_size;
  HloModule* parent_module;

 public:
  explicit TensorSplitterRewriteVisitor(int64_t max_size_threshold,
                                        int64_t target_split_size,
                                        HloModule* parent_module)
      : max_size_threshold(max_size_threshold),
        target_split_size(target_split_size),
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
                         std::vector<int64_t>* original_dimensions = nullptr,
                         std::vector<int64_t>* exclude_dimensions = nullptr);

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
  int64_t BestSplitDim(HloInstruction* inst,
                       absl::Span<const int64_t> excluded);

  // Given a split dimension, determine the best possible split
  // size with equally shaped pieces. If no split size is possible, returns -1.
  int64_t BestEvenSplitSize(HloInstruction* inst, int64_t split_dim);

  // Given a split dimension, determine the best possible split
  // size, allowing for un uneven split. Split_count denotes the
  // number of pieces of split_size size; split_rest is the size of
  // the last piece.
  SplitProperties DetermineSplitSize(HloInstruction* inst, int64_t split_dim);

  HloComputation* CreateWhileSplitCondition(const std::string& name,
                                            const Shape& parameters_shape,
                                            int64_t stop_at);

  std::vector<HloInstruction*> CreateWhileSplitWithResults(
      HloComputation* parent_comp, HloComputation* condition,
      HloComputation* body, std::vector<HloInstruction*> parameters,
      std::vector<std::tuple<int64_t, Shape>> ids_and_shapes);

  Status HandleDot(HloInstruction* dot) override;

  Status HandleReduce(HloInstruction* reduce) override;

  Status HandleSort(HloInstruction* sort) override;

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

    using VisitedInstructionKey = std::tuple<int, int64_t, int64_t>;
    absl::flat_hash_map<VisitedInstructionKey, HloInstruction*>
        visited_instructions_;

    HloComputation::Builder& builder_;
    absl::Span<HloInstruction*> leafs_;

   public:
    explicit Splitter(HloComputation::Builder& builder, HloComputation* parent,
                      absl::Span<HloInstruction*> leafs, int64_t offset = 0)
        : builder_(builder), leafs_(leafs) {
      std::stringstream msg;

      msg << "@@@ leafs=[";
      for (auto leaf : leafs_) {
        msg << leaf->name() << ",";
      }
      msg << "]";
      LOG(INFO) << "\n> " << msg.str();

      HloInstruction* init_offset =
          parent->AddInstruction(CONSTANT_INT64(offset));
      parameters_.push_back(init_offset);

      // Make a param, the shape can be added to over time to get correct
      // shape
      Shape param_shape = ShapeUtil::MakeTupleShape({init_offset->shape()});
      param_ = builder.AddInstruction(
          HloInstruction::CreateParameter(0, param_shape, "loop_param"));
      offset_ = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          init_offset->shape(), param_, 0));
    }

    StatusOr<HloInstruction*> SplitInstruction(HloInstruction* inst,
                                               int64_t split_dim,
                                               int64_t split_size);

    StatusOr<HloInstruction*> SplitLeafDot(HloInstruction* dot,
                                           int64_t split_dim,
                                           int64_t split_size);

    StatusOr<HloInstruction*> SplitLeafBroadcast(HloInstruction* broadcast,
                                                 int64_t split_dim,
                                                 int64_t split_size);

    StatusOr<HloInstruction*> SplitLeafParameter(HloInstruction* parameter,
                                                 int64_t split_dim,
                                                 int64_t split_size);

    StatusOr<HloInstruction*> SplitLeafIota(HloInstruction* iota,
                                            int64_t split_dim,
                                            int64_t split_size);

    VisitedInstructionKey MakeVisitedInstructionKey(const HloInstruction* inst,
                                                    int64_t split_dim,
                                                    int64_t split_size) {
      int unique_id = inst->unique_id();
      return std::make_tuple(unique_id, split_dim, split_size);
    }

    // Add the parameter and returnd it's index in the tuple. If get_tuple
    // is passed, it will also create an accessor for the parameter.
    int64_t AddParameter(HloInstruction* inst,
                         HloInstruction** get_tuple = nullptr) {
      int64_t idx = parameters_size();
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
    HloInstruction* BuildOutputTuple(int64_t split_dim, int64_t split_size,
                                     HloInstruction* original,
                                     HloInstruction* part,
                                     bool combine_with_sum = false,
                                     bool combine_with_reduce = false);

    int64_t parameters_size() { return parameters_.size(); }

    HloInstruction* parameters(int64_t idx) { return parameters_.at(idx); }

    HloInstruction* tuple_parameters() { return param_; }

    std::vector<HloInstruction*>& parameters() { return parameters_; }

    HloInstruction* offset() { return offset_; }
  };
};

}  // namespace

bool TensorSplitterRewriteVisitor::OperandShouldBeSplit(HloInstruction* inst) {
  if (!inst->shape().IsArray()) {
    return false;
  }
  return ShapeUtil::ByteSizeOfElements(inst->shape()) > max_size_threshold;
}

bool TensorSplitterRewriteVisitor::OperandCanBeSplit(
    HloInstruction* inst, std::vector<HloInstruction*>* split_leafs,
    std::vector<int64_t>* original_dimensions,
    std::vector<int64_t>* exclude_dimensions) {
  std::stringstream msg;
  msg << "\n<---><---><---><---> orig=[";
  for (auto dim : *original_dimensions) {
    msg << dim << ",";
  }
  msg << "], excl=[";
  for (auto dim : *exclude_dimensions) {
    msg << dim << ",";
  }
  msg << "]";
  msg << "\n> Can be split for '" << inst->name() << "'";
  HloInstruction *next, *lhs, *rhs;
  std::vector<HloInstruction*> next_vec;
  if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
    bool do_split_lhs = OperandShouldBeSplit(lhs);
    bool do_split_rhs = OperandShouldBeSplit(rhs);

    msg << ", do_split_lhs=" << do_split_lhs;
    msg << ", do_split_rhs=" << do_split_rhs;

    if (do_split_lhs && do_split_rhs) {
      // We can only split one dimension, so this is impossible
      LOG(INFO) << msg.str();
      return false;
    } else if (do_split_lhs) {
      msg << ", split LHS;";
      LOG(INFO) << msg.str();
      // Exclude all rhs dims from split
      for (int64_t i = lhs->shape().dimensions_size() - 1;
           i < original_dimensions->size(); i++) {
        exclude_dimensions->push_back((*original_dimensions)[i]);
      }
      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[dnums.lhs_batch_dimensions(i)]);
      }
      // Make a dimensions which is only for the lhs
      std::vector<int64_t> lhs_original_dims;
      int64_t lhs_cdim =
          inst->dot_dimension_numbers().lhs_contracting_dimensions(0);
      for (int64_t i = 0; i < lhs->shape().dimensions_size(); i++) {
        if (i == lhs_cdim) {
          lhs_original_dims.push_back(-1);  // this has no original dim...
        } else if (i < lhs_cdim) {
          lhs_original_dims.push_back((*original_dimensions)[i]);
        } else if (i > lhs_cdim) {
          lhs_original_dims.push_back((*original_dimensions)[i - 1]);
        }
      }
      // Check if can split
      bool can_split = OperandCanBeSplit(lhs, split_leafs, &lhs_original_dims,
                                         exclude_dimensions);
      lhs_original_dims.clear();
      return can_split;  // not tail recursive to keep fresh orig dims
    } else if (do_split_rhs) {
      msg << ", split RHS;";
      LOG(INFO) << msg.str();
      // Exclude all lhs dims from split
      for (int64_t i = 0; i < lhs->shape().dimensions_size() - 1; i++) {
        exclude_dimensions->push_back((*original_dimensions)[i]);
      }
      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      int64_t offset = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < dnums.rhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)]);
      }
      // Make a dimensions which is only for the rhs
      std::vector<int64_t> rhs_original_dims;
      int64_t rhs_cdim =
          inst->dot_dimension_numbers().rhs_contracting_dimensions(0);
      int64_t rhs_start = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < rhs->shape().dimensions_size(); i++) {
        if (i == rhs_cdim) {
          rhs_original_dims.push_back(-1);  // this has no original dim...
        } else if (i < rhs_cdim) {
          rhs_original_dims.push_back((*original_dimensions)[rhs_start + i]);
        } else if (i > rhs_cdim) {
          rhs_original_dims.push_back(
              (*original_dimensions)[rhs_start + i - 1]);
        }
      }
      // Check if can split
      bool can_split = OperandCanBeSplit(rhs, split_leafs, &rhs_original_dims,
                                         exclude_dimensions);
      rhs_original_dims.clear();
      return can_split;  // not tail recursive to keep fresh orig dims
    } else {
      msg << ", dot base case;";
      LOG(INFO) << msg.str();
      // Base case: A Dot produces this large intermediate tensor

      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[dnums.lhs_batch_dimensions(i)]);
      }
      int64_t offset = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < dnums.rhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)]);
      }
      if (split_leafs != nullptr) {
        split_leafs->push_back(inst);
      }
      return true;
    }
  } else if (Match(inst, m::Broadcast(m::Op()))) {
    LOG(INFO) << msg.str();
    // Base case: A broadcast can be split
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
  } else if (Match(inst, m::Iota())) {
    LOG(INFO) << msg.str();
    // Base case: An Iota can be split!
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
  } else if (Match(inst, m::Parameter())) {
    LOG(INFO) << msg.str();
    LOG(INFO) << "\n>---<>---<  Exit. Parameter will be split '" << inst->name()
              << "'";
    // TODO(awav)
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
    // TODO(awav)
    // return false;
  } else if (Match(inst, m::Transpose(m::Op(&next)))) {
    // A transpose changes the dimensions, so we need to
    // update the original_dimensions array.

    // if (original_dimensions != nullptr) {
    //   std::vector<int64_t>
    //   old_original_dimensions(original_dimensions->begin(),
    //                                                original_dimensions->end());
    //   for (int64_t i = 0; i < original_dimensions->size(); i++) {
    //     (*original_dimensions)[i] =
    //         old_original_dimensions[inst->dimensions(i)];
    //   }
    // }
    // return OperandCanBeSplit(next, split_leafs, original_dimensions,
    //                          exclude_dimensions);

    if (original_dimensions == nullptr) {
      LOG(INFO) << msg.str();
      return OperandCanBeSplit(next, split_leafs, original_dimensions,
                               exclude_dimensions);
    }

    msg << ", transpose original dims to [";
    std::vector<int64_t> transposed_dimensions(original_dimensions->begin(),
                                               original_dimensions->end());
    for (int64_t i = 0; i < original_dimensions->size(); i++) {
      transposed_dimensions[i] = original_dimensions->at(inst->dimensions(i));
      msg << transposed_dimensions[i] << ",";
    }
    msg << "]";
    LOG(INFO) << msg.str();
    return OperandCanBeSplit(next, split_leafs, &transposed_dimensions,
                             exclude_dimensions);
  } else if (MatchSupportedNestedReduce(inst)) {
    msg << ", split nested reduce;";
    LOG(INFO) << msg.str();
    return OperandCanBeSplit(inst->mutable_operand(0), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
    msg << ", split triangular solve;";
    LOG(INFO) << msg.str();
    // We can split a triangular solve on some (but not all)
    // dims
    if (original_dimensions != nullptr && exclude_dimensions != nullptr) {
      if (inst->triangular_solve_options().left_side()) {
        // exclude second to last : Ax = y
        exclude_dimensions->push_back(
            original_dimensions->at(original_dimensions->size() - 2));
      } else {
        // exclude last : xA = y
        exclude_dimensions->push_back(
            original_dimensions->at(original_dimensions->size() - 1));
      }
    }
    // We can't split the matrix for now, so ignore it
    return OperandCanBeSplit(inst->mutable_operand(1), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (MatchPointwiseUnary(inst, &next)) {
    msg << ", split pointwise unary;";
    LOG(INFO) << msg.str();
    // This is a special case seperate from nary,
    // since we can make it tail recursive :)
    return OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions);
  } else if (MatchPointwiseNary(inst, &next_vec)) {
    msg << ", split pointwise nary;";
    LOG(INFO) << msg.str();

    for (HloInstruction* next : next_vec) {
      // this path is not tail recursive :(
      if (!OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions)) {
        LOG(INFO) << "\n>---<>---<  Exit. Cannot be split 1 for '"
                  << next->name() << "'";
        ;
        return false;
      }
    }
    return true;
  } else {
    LOG(INFO) << msg.str();
    LOG(INFO) << "\n>---<>---<  Exit. Cannot be split 0 for '" << inst->name()
              << "'";
    return false;
  }
}

bool TensorSplitterRewriteVisitor::MatchPointwiseUnary(
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

bool TensorSplitterRewriteVisitor::MatchPointwiseNary(
    HloInstruction* inst, std::vector<HloInstruction*>* operands) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() > 0) {
    if (operands != nullptr) {
      for (int64_t i = 0; i < inst->operand_count(); i++)
        operands->push_back(inst->mutable_operand(i));
    }
    return true;
  } else {
    return false;
  }
}

bool TensorSplitterRewriteVisitor::MatchSupportedReduce(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReduce) {
    int64_t opt_count = inst->operand_count() / 2;
    if (opt_count < 1) return false;

    for (int64_t i = 1; i < opt_count; i++)
      if (!ShapeUtil::EqualIgnoringElementType(inst->operand(0)->shape(),
                                               inst->operand(i)->shape()))
        return false;

    for (int64_t i = 0; i < opt_count; i++)
      if (!ShapeUtil::IsScalar(inst->operand(opt_count + i)->shape()))
        return false;

    return true;
  } else {
    return false;
  }
}

bool TensorSplitterRewriteVisitor::MatchSupportedNestedReduce(
    HloInstruction* inst) {
  return MatchSupportedReduce(inst) && inst->operand_count() == 2;
}

int64_t TensorSplitterRewriteVisitor::BestSplitDim(
    HloInstruction* inst, absl::Span<const int64_t> excluded) {
  const Shape& shape = inst->shape();
  int64_t best_dim = -1,
          best_split = 0;  // ShapeUtil::ElementsIn(inst->shape());
  for (int64_t i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) {
      continue;
    }
    int64_t split = BestEvenSplitSize(inst, i);
    if (split == -1 || split <= best_split) {
      continue;
    }
    best_split = split;
    best_dim = i;
  }
  return best_dim;
}

std::vector<HloInstruction*> CreateStartIndicesArray(
    HloComputation::Builder& builder, int64_t dimensions_size, int64_t init,
    int64_t changed_dimension, HloInstruction* changed_instruction) {
  std::vector<HloInstruction*> start_indices;
  for (auto i = 0; i < dimensions_size; i++) {
    if (i == changed_dimension) {
      start_indices.push_back(changed_instruction);
    } else {
      start_indices.push_back(builder.AddInstruction(CONSTANT_INT64(init)));
    }
  }
  return start_indices;
}

#define PRIME_LEN 512
const int64_t primes[PRIME_LEN] = {
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069,
    2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
    2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
    2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347,
    2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
    2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543,
    2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657,
    2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
    2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801,
    2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
    2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011,
    3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
    3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221,
    3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
    3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413,
    3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527,
    3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607,
    3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671};

int64_t BestEvenSplitSizeFold(int64_t (&factors)[PRIME_LEN], int offset,
                              int64_t current, int64_t best, int64_t size,
                              int64_t max_size) {
  if (offset >= PRIME_LEN) {
    return best;
  } else {
    if (factors[offset] > 0) {
      int64_t current_prime = primes[offset] * current;
      if (size / current_prime <= max_size && current_prime < best) {
        best = current_prime;
      }
      factors[offset]--;
      best = BestEvenSplitSizeFold(factors, offset, current_prime, best, size,
                                   max_size);
      factors[offset]++;
    }
    return BestEvenSplitSizeFold(factors, offset + 1, current, best, size,
                                 max_size);
  }
}

int64_t TensorSplitterRewriteVisitor::BestEvenSplitSize(HloInstruction* inst,
                                                        int64_t split_dim) {
  // find list of prime factors
  int64_t factors[PRIME_LEN];
  int64_t tmp_size = inst->shape().dimensions(split_dim);
  for (int i = 0; i < PRIME_LEN; i++) {
    factors[i] = 0;
    while (tmp_size % primes[i] == 0) {
      factors[i]++;
      tmp_size /= primes[i];
    }
  }

  int64_t size = inst->shape().dimensions(split_dim);
  int64_t full_size_bytes =
      ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type()) *
      ShapeUtil::ElementsIn(inst->shape());
  int64_t max_size = target_split_size * size / full_size_bytes;
  int64_t factor = BestEvenSplitSizeFold(factors, 0, 1, size, size, max_size);
  return size / factor;
}

SplitProperties TensorSplitterRewriteVisitor::DetermineSplitSize(
    HloInstruction* inst, int64_t split_dim) {
  const Shape& inst_shape = inst->shape();
  int64_t best_even_split = BestEvenSplitSize(inst, split_dim);
  int64_t split_dim_size = inst_shape.dimensions(split_dim);
  int64_t primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(inst_shape.element_type());
  int64_t max_elements = target_split_size / primitive_type_size;
  int64_t max_dim_size =
      max_elements * split_dim_size / ShapeUtil::ElementsIn(inst_shape);

  int64_t size, count, rest;
  if (best_even_split >= max_dim_size * 8 / 10) {
    // even split is prefered
    size = best_even_split;
    count = split_dim_size / size;
    rest = 0;
  } else {
    // uneven split is prefered
    size = max_dim_size;
    count = split_dim_size / size;
    rest = split_dim_size - size * count;
  }
  return std::make_tuple(size, count, rest);
}

StatusOr<HloInstruction*>
TensorSplitterRewriteVisitor::Splitter::SplitInstruction(HloInstruction* inst,
                                                         int64_t split_dim,
                                                         int64_t split_size) {
  LOG(INFO) << "\n @@@ Enter SplitInstruction for '" << inst->name() << "'";
  auto visited_inst_key =
      MakeVisitedInstructionKey(inst, split_dim, split_size);
  if (visited_instructions_.contains(visited_inst_key)) {
    LOG(INFO) << "\n &&& Found a duplicate for " << inst->name() << ", "
              << split_dim << ", " << split_size << ">";
    return visited_instructions_[visited_inst_key];
  }

  if (absl::c_linear_search(leafs_, inst)) {
    LOG(INFO) << "\n> Found in leafs '" << inst->name() << "'";
    if (Match(inst, m::Dot())) {
      LOG(INFO) << "\n# Split 'Dot' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafDot(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Broadcast())) {
      LOG(INFO) << "\n# Split 'Broadcast' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafBroadcast(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Parameter())) {
      LOG(INFO) << "\n# Split 'Parameter' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafParameter(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Iota())) {
      LOG(INFO) << "\n# Split 'Iota' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafIota(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    }
  } else {
    HloInstruction *operand, *lhs, *rhs;
    std::vector<HloInstruction*> operands;

    if (Match(inst, m::Transpose(m::Op(&operand)))) {
      // For a transpose, the transpose might change which dimension is
      // being split. So we obtain the new split dimension and then
      // recursively a new operand to make a clone.
      int64_t operand_split_dim = inst->dimensions(split_dim);
      LOG(INFO) << "\n# Split 'Transpose:Op' instruction '" << inst->name()
                << "'";
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(operand, operand_split_dim, split_size));

      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst = builder_.AddInstruction(
          inst->CloneWithNewOperands(new_shape, {new_operand}));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (MatchSupportedNestedReduce(inst)) {
      // For a reduce, split the 0th and only operand
      // (the initializer a scalar, so all we need to do
      // is update the shape and clone the operand with new
      // inputs)

      LOG(INFO) << "\n# Split 'NestedReduce' instruction '" << inst->name()
                << "'";
      int64_t operand_split_dim = split_dim;  // split dim in operand
      if (inst->dimensions(0) <= split_dim) {
        operand_split_dim += 1;
      }

      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                          SplitInstruction(inst->mutable_operand(0),
                                           operand_split_dim, split_size));

      HloInstruction* init_operand = inst->mutable_operand(1);
      HloInstruction* new_init_operand;
      AddParameter(init_operand, &new_init_operand);

      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst =
          builder_.AddInstruction(inst->CloneWithNewOperands(
              new_shape, {new_operand, new_init_operand}));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
      LOG(INFO) << "\n# Split 'TriangularSolve' instruction '" << inst->name()
                << "'";
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(inst->mutable_operand(1), split_dim, split_size));
      HloInstruction* mat;
      AddParameter(inst->mutable_operand(0), &mat);
      HloInstruction* new_inst = builder_.AddInstruction(
          inst->CloneWithNewOperands(new_operand->shape(), {mat, new_operand}));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
      std::stringstream msg;
      msg << "\n# Split 'Dot(Op, Op)' instruction '" << inst->name() << "'";
      // For an intermediate dot, split the correct operand and assemble
      // a new dot.
      bool split_lhs = ShapeUtil::ElementsIn(lhs->shape()) >
                       ShapeUtil::ElementsIn(
                           rhs->shape());  // this works, since only one of the
                                           // operands needs to be split
      msg << ", split_dim=" << split_dim << ", split_size=" << split_size
          << ", is_lhs=" << (split_lhs ? "yes" : "no");
      LOG(INFO) << msg.str();

      if (split_lhs) {
        CHECK(split_dim < lhs->shape().dimensions_size() - 1);
        int64_t lhs_contr_dim =
            inst->dot_dimension_numbers().lhs_contracting_dimensions(0);
        int64_t lhs_split_dim =
            split_dim >= lhs_contr_dim ? split_dim + 1 : split_dim;

        TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                            SplitInstruction(lhs, lhs_split_dim, split_size));
        HloInstruction* param_rhs;
        AddParameter(rhs, &param_rhs);

        Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                               inst->shape().dimensions());
        new_shape.set_dimensions(split_dim, split_size);
        HloInstruction* new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(new_shape, {new_lhs, param_rhs}));
        visited_instructions_[visited_inst_key] = new_inst;
        return new_inst;
      } else {
        int64_t rhs_start = lhs->shape().dimensions_size() - 1;
        CHECK(split_dim >= rhs_start);
        int64_t rhs_contr_dim =
            inst->dot_dimension_numbers().rhs_contracting_dimensions(0);
        int64_t rhs_split_dim = split_dim - rhs_start >= rhs_contr_dim
                                    ? split_dim + 1 - rhs_start
                                    : split_dim - rhs_start;

        TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                            SplitInstruction(rhs, rhs_split_dim, split_size));
        Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                               inst->shape().dimensions());
        HloInstruction* param_lhs;
        AddParameter(lhs, &param_lhs);

        new_shape.set_dimensions(split_dim, split_size);
        HloInstruction* new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(new_shape, {param_lhs, new_rhs}));
        visited_instructions_[visited_inst_key] = new_inst;
        return new_inst;
      }
    } else if (MatchPointwiseNary(inst, &operands)) {
      // For a pointwise operation recursively obtain the new operands and
      // clone the operation.
      LOG(INFO) << "\n# Split 'PointwiseNary' instruction '" << inst->name()
                << "'";
      std::vector<HloInstruction*> ops;
      for (HloInstruction* operand : operands) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                            SplitInstruction(operand, split_dim, split_size));
        ops.push_back(new_operand);
      }
      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst = builder_.AddInstruction(
          inst->CloneWithNewOperands(new_shape, absl::MakeSpan(ops)));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else {
      // Invariant violation
      // TODO: Is there a more idiomatic way to return a bad status?
      LOG(ERROR) << "Trying to split invalid operation '" << inst->name()
                 << "'";
      CHECK(false);
    }
  }
}

StatusOr<HloInstruction*> TensorSplitterRewriteVisitor::Splitter::SplitLeafDot(
    HloInstruction* dot, int64_t split_dim, int64_t split_size) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // For the dot we identify the parameter to split and then
  // Generate the final dot operation, as well as the operand
  // vector.

  Shape dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                         dot->shape().dimensions());
  dot_shape.set_dimensions(split_dim, split_size);

  auto& dnums = dot->dot_dimension_numbers();
  int64_t dims_lhs =
      lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();

  HloInstruction *split_op, *join_op;
  bool split_is_lhs;
  if (split_dim < dims_lhs) {
    // We are splitting up the lhs
    split_is_lhs = true;
    split_op = lhs;
    join_op = rhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.lhs_contracting_dimensions(i)) split_dim += 1;
    }
  } else {
    // We are splitting up the rhs
    split_is_lhs = false;
    split_dim -= dims_lhs;
    split_op = rhs;
    join_op = lhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64_t i = 0; i < dnums.rhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.rhs_contracting_dimensions(i)) split_dim += 1;
    }
  }

  LOG(INFO) << "<<< "
            << "Splitting leaf dot " << dot->name()
            << "; split_dim=" << split_dim << "; split_size=" << split_size
            << "; split_lhs=" << (split_is_lhs ? "yes" : "no");

  // add parameters
  HloInstruction* split_op_param;
  int64_t split_op_tuple_idx = AddParameter(split_op, &split_op_param);
  HloInstruction* join_op_param;
  int64_t join_op_tuple_idx = AddParameter(join_op, &join_op_param);

  // dynamic slice by index
  Shape split_shape = ShapeUtil::MakeShape(split_op->shape().element_type(),
                                           split_op->shape().dimensions());
  split_shape.set_dimensions(split_dim, split_size);

  std::vector<HloInstruction*> start_indices;
  for (int64_t dim = 0; dim < split_shape.dimensions_size(); dim++) {
    if (dim == split_dim) {
      start_indices.push_back(offset_);
    } else {
      start_indices.push_back(builder_.AddInstruction(CONSTANT_INT64(0)));
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
TensorSplitterRewriteVisitor::Splitter::SplitLeafBroadcast(
    HloInstruction* broadcast, int64_t split_dim, int64_t split_size) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));

  // For a broadcast, we identify if we can split it by
  // changeing the broadcast itself, of if we have to
  // create slices of the underlying operand tensor.

  bool split_on_original_dim =
      absl::c_linear_search(broadcast->dimensions(), split_dim);

  int64_t parameter_idx;
  Shape parameter_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                               operand->shape().dimensions());

  std::stringstream msg;
  msg << "broadcast->dimentions[";
  for (auto d : broadcast->dimensions()) msg << d << ",";
  msg << "], broadcast->dimentions().size=" << broadcast->dimensions().size();
  msg << ", split_dim=" << split_dim << ", split_size=" << split_size;

  msg << ", split_on_original_dim=" << split_on_original_dim;
  msg << ", operand_shape=" << parameter_shape;
  LOG(INFO) << "\n> @@@ " << msg.str();

  HloInstruction* new_operand;

  if (split_on_original_dim) {
    // we need to slice the parameter ...
    int64_t operand_split_dim;
    for (int64_t i = 0; i < broadcast->dimensions().size(); i++) {
      if (broadcast->dimensions(i) == split_dim) {
        operand_split_dim = i;
        break;
      }
    }

    parameter_shape.set_dimensions(operand_split_dim, split_size);

    std::vector<HloInstruction*> start_indices;
    for (int64_t dim = 0; dim < operand->shape().dimensions_size(); dim++) {
      if (dim == operand_split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(builder_.AddInstruction(CONSTANT_INT64(0)));
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
TensorSplitterRewriteVisitor::Splitter::SplitLeafParameter(
    HloInstruction* parameter, int64_t split_dim, int64_t split_size) {
  CHECK(Match(parameter, m::Parameter()));
  const Shape& parameter_shape = parameter->shape();
  const auto& parameter_dims = parameter_shape.dimensions();
  const auto& element_type = parameter_shape.element_type();
  CHECK(parameter_shape.dimensions_size() > split_dim);

  HloInstruction* get_tuple_parameter;
  auto parameter_idx = AddParameter(parameter, &get_tuple_parameter);

  Shape slice_shape = ShapeUtil::MakeShape(element_type, parameter_dims);
  slice_shape.set_dimensions(split_dim, split_size);

  std::vector<HloInstruction*> start_indices;
  for (auto dim = 0; dim < parameter_shape.dimensions_size(); dim++) {
    if (dim == split_dim) {
      start_indices.push_back(offset_);
    } else {
      start_indices.push_back(builder_.AddInstruction(CONSTANT_INT64(0)));
    }
  }

  std::stringstream msg;
  msg << "parameter '" << parameter->name() << "' dimensions[";
  for (auto d : parameter_dims) msg << d << ",";
  msg << "], parameter->dimentions().size=" << parameter_dims.size();
  msg << ", shape=" << parameter_shape;
  msg << ", split_dim=" << split_dim << ", split_size=" << split_size;
  msg << ", split_dim_for_param=" << parameter_shape.dimensions(split_dim);
  LOG(INFO) << "\n> @@@ " << msg.str();

  return builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
      slice_shape, get_tuple_parameter, absl::MakeSpan(start_indices),
      slice_shape.dimensions()));
}

StatusOr<HloInstruction*> TensorSplitterRewriteVisitor::Splitter::SplitLeafIota(
    HloInstruction* iota, int64_t split_dim, int64_t split_size) {
  CHECK(Match(iota, m::Iota()));

  // For an iota, we simply produce smaller iota and add the
  // loop offset to each parameter

  auto* iota_inst = DynCast<HloIotaInstruction>(iota);
  CHECK(iota_inst != nullptr);

  int64_t parameter_idx = 0;
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

    std::vector<int64_t> broadcast_dims = {};
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

HloInstruction* TensorSplitterRewriteVisitor::Splitter::BuildOutputTuple(
    int64_t split_dim, int64_t split_size, HloInstruction* original,
    HloInstruction* part, bool combine_with_sum, bool combine_with_reduce) {
  HloInstruction* output;
  int64_t output_idx;
  if (combine_with_reduce) {
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    // re-use reduce init for output init
    HloInstruction* output_init = original->mutable_operand(1);
    if (!ShapeUtil::IsScalar(original->shape())) {
      CHECK(ShapeUtil::IsScalar(output_init->shape()));
      output_init = original->parent()->AddInstruction(
          HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
    }
    output_idx = AddParameter(output_init, &output);
  } else {
    // create the output init (broadcast off of 0)
    HloInstruction* output_init =
        original->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(original->shape().element_type())));
    output_init = original->parent()->AddInstruction(
        HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
    output_idx = AddParameter(output_init, &output);
  }

  HloInstruction* updated_output;
  if (combine_with_sum) {
    // we're splitting a dot on a dot dimension, this means
    // all that needs to be done is adding the part onto the
    // result (which is initialized as 0)
    updated_output = builder_.AddInstruction(HloInstruction::CreateBinary(
        output->shape(), HloOpcode::kAdd, output, part));
  } else if (combine_with_reduce) {
    // we're splitting on a reduced dimension
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    HloComputation* reduce_fn = original->called_computations()[0];
    if (ShapeUtil::IsScalar(output->shape())) {
      // we can call the function directly
      updated_output = builder_.AddInstruction(HloInstruction::CreateCall(
          original->shape(), {output, part}, reduce_fn));
    } else {
      // we have to call the function through map
      updated_output = builder_.AddInstruction(HloInstruction::CreateMap(
          original->shape(), {output, part}, reduce_fn));
    }
  } else {
    // slice part onto output
    std::vector<HloInstruction*> start_indices;
    for (int64_t dim = 0; dim < output->shape().dimensions_size(); dim++) {
      if (dim == split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(builder_.AddInstruction(CONSTANT_INT64(0)));
      }
    }
    updated_output =
        builder_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            output->shape(), output, part, start_indices));
  }

  // add split size to index
  HloInstruction* split_size_const =
      builder_.AddInstruction(CONSTANT_INT64(split_size));
  HloInstruction* updated_index =
      builder_.AddInstruction(HloInstruction::CreateBinary(
          offset_->shape(), HloOpcode::kAdd, offset_, split_size_const));

  // collect idx, output and all parameters into a tuple ..
  std::vector<HloInstruction*> output_elements = {updated_index};
  for (int64_t i = 1; i < param_->shape().tuple_shapes_size(); i++) {
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

HloComputation* TensorSplitterRewriteVisitor::CreateWhileSplitCondition(
    const std::string& name, const Shape& parameters_shape, int64_t stop_at) {
  HloComputation::Builder builder(name);
  HloInstruction* parameter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, parameters_shape, "loop_param"));
  HloInstruction* iteration =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          parameters_shape.tuple_shapes(0), parameter, 0));
  HloInstruction* stop_iteration =
      builder.AddInstruction(CONSTANT_INT64(stop_at));
  HloInstruction* compare = builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), iteration,
                                    stop_iteration, ComparisonDirection::kLt));
  return parent_module->AddEmbeddedComputation(builder.Build(compare));
}

std::vector<HloInstruction*>
TensorSplitterRewriteVisitor::CreateWhileSplitWithResults(
    HloComputation* parent_comp, HloComputation* condition,
    HloComputation* body, std::vector<HloInstruction*> parameters,
    std::vector<std::tuple<int64_t, Shape>> ids_and_shapes) {
  // Create init and loop
  HloInstruction* init =
      parent_comp->AddInstruction(HloInstruction::CreateTuple(parameters));
  HloInstruction* loop = parent_comp->AddInstruction(
      HloInstruction::CreateWhile(init->shape(), condition, body, init));

  // Extract While results for array and indices
  std::vector<HloInstruction*> results;
  for (auto id_shape : ids_and_shapes) {
    int64_t id;
    Shape shape;
    std::tie(id, shape) = id_shape;
    HloInstruction* result = parent_comp->AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, loop, id));
    results.push_back(result);
  }
  return results;
}

Status TensorSplitterRewriteVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  std::stringstream ss;

  LOG(INFO) << "\n ---------------------------"
            << "\n ----> Enter HandleDot for '" << dot->name() << "'";

  if (OperandShouldBeSplit(dot)) {
    ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return Status::OK();
  }

  // TODO: Handle the case where both operands can be
  //       split in a better way.

  // Cases we handle:
  // 1. lhs can split
  // 2. rhs can split
  // 3. lhs = rhs + can split
  // 3.1 lhs + rhs can split on respective contracted dim

  bool can_split = false;
  bool rhs_is_lhs = lhs == rhs;

  std::vector<HloInstruction*> split_leafs_lhs;
  std::vector<HloInstruction*> split_leafs_rhs;
  std::vector<int64_t> exclude_dims_lhs;
  std::vector<int64_t> exclude_dims_rhs;

  auto lhs_dim_size = lhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_lhs(lhs_dim_size);
  std::iota(original_dims_lhs.begin(), original_dims_lhs.end(), 0);

  bool can_split_lhs = OperandShouldBeSplit(lhs) &&
                       OperandCanBeSplit(lhs, &split_leafs_lhs,
                                         &original_dims_lhs, &exclude_dims_lhs);

  auto rhs_dim_size = rhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_rhs(rhs_dim_size);
  std::iota(original_dims_rhs.begin(), original_dims_rhs.end(), 0);

  bool can_split_rhs = OperandShouldBeSplit(rhs) &&
                       OperandCanBeSplit(rhs, &split_leafs_rhs,
                                         &original_dims_rhs, &exclude_dims_rhs);

  if (can_split_lhs && can_split_rhs && rhs_is_lhs) {
    //
    // Case :: Self dot
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);
    if (dnums.lhs_contracting_dimensions()[0] !=
        dnums.rhs_contracting_dimensions()[0]) {
      return Status::OK();
    }

    int64_t split_dim = dnums.lhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_lhs, split_dim)) {
      LOG(WARNING) << "Failed to split self dot '" << dot->name()
                   << "' as contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return Status::OK();
    }

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest == lhs->shape().dimensions(split_dim));

    ss << "<<< "
       << "Splitting self dot " << dot->name()
       << " operand will be split at dimension " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    HloComputation::Builder body_builder("tensor_splitter_dot_body");
    Splitter splitter(body_builder, dot->parent(),
                      absl::MakeSpan(split_leafs_lhs));

    TF_ASSIGN_OR_RETURN(HloInstruction * split_lhs,
                        splitter.SplitInstruction(lhs, split_dim, split_size));

    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(dot->shape(), {split_lhs, split_lhs}));

    HloInstruction* output_tuple = splitter.BuildOutputTuple(
        -1, split_size, dot, part, /*combine_with_sum =*/true);
    HloComputation* body =
        parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));

    // build the condition
    HloComputation::Builder cond_builder("tensor_splitter_dot_cond");
    HloInstruction* cond_param =
        cond_builder.AddInstruction(HloInstruction::CreateParameter(
            0, output_tuple->shape(), "loop_param"));
    HloInstruction* cond_offset =
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_tuple->shape().tuple_shapes(0), cond_param, 0));
    HloInstruction* offset_less_than =
        cond_builder.AddInstruction(CONSTANT_INT64(main_split_size));
    HloInstruction* compare =
        cond_builder.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
            ComparisonDirection::kLt));
    HloComputation* cond =
        parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

    // build the while and replace the original element with a get tuple.
    int64_t output_idx = output_tuple->shape().tuple_shapes().size() - 1;
    HloInstruction* init = dot->parent()->AddInstruction(
        HloInstruction::CreateTuple(splitter.parameters()));
    HloInstruction* loop = dot->parent()->AddInstruction(
        HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
    HloInstruction* result = dot->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(dot->shape(), loop, output_idx));

    if (split_rest == 0) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      return ReplaceInstruction(dot, result);
    } else {
      HloComputation::Builder rest_builder("tensor_splitter_dot_rest");
      Splitter splitter(rest_builder, dot->parent(),
                        absl::MakeSpan(split_leafs_lhs), main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_lhs,
          splitter.SplitInstruction(lhs, split_dim, split_rest));

      HloInstruction* rest_part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), {rest_lhs, rest_lhs}));

      HloComputation* rest_body =
          parent_module->AddEmbeddedComputation(rest_builder.Build(rest_part));

      splitter.parameters()[0] =
          dot->parent()->AddInstruction(CONSTANT_INT64(main_split_size));
      HloInstruction* args = dot->parent()->AddInstruction(
          HloInstruction::CreateTuple(splitter.parameters()));
      HloInstruction* rest_result = dot->parent()->AddInstruction(
          HloInstruction::CreateCall(rest_part->shape(), {args}, rest_body));

      HloInstruction* full_result =
          dot->parent()->AddInstruction(HloInstruction::CreateBinary(
              result->shape(), HloOpcode::kAdd, result, rest_result));
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      return ReplaceInstruction(dot, full_result);
    }
  } else if (!rhs_is_lhs && can_split_lhs && can_split_rhs) {
    //
    // CASE :: both lhs and rhs need split
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);

    int64_t split_dim_lhs = dnums.lhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_lhs, split_dim_lhs)) {
      LOG(WARNING) << "Failed to split both sides of dot '" << dot->name()
                   << "' as LHS contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      return Status::OK();
    }
    int64_t split_dim_rhs = dnums.rhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_rhs, split_dim_rhs)) {
      LOG(WARNING) << "Failed to split both sides of dot '" << dot->name()
                   << "' as RHS contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return Status::OK();
    }

    CHECK(lhs->shape().dimensions(split_dim_lhs) ==
          rhs->shape().dimensions(split_dim_rhs));

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim_lhs);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          lhs->shape().dimensions(split_dim_lhs));

    ss << "<<< "
       << "Splitting dot " << dot->name()
       << " lhs and rhs will be split on contracted dimension with split size "
       << split_size << " and split rest " << split_rest;

    HloComputation::Builder body_builder("tensor_splitter_dot_body");

    for (HloInstruction* leaf : split_leafs_rhs) {
      split_leafs_lhs.push_back(leaf);
    }

    Splitter splitter(body_builder, dot->parent(),
                      absl::MakeSpan(split_leafs_lhs));

    ss << "\n> Split LHS '" << lhs->name() << "'";
    TF_ASSIGN_OR_RETURN(
        HloInstruction * split_lhs,
        splitter.SplitInstruction(lhs, split_dim_lhs, split_size));

    ss << "\n> Split RHS '" << rhs->name() << "'";
    TF_ASSIGN_OR_RETURN(
        HloInstruction * split_rhs,
        splitter.SplitInstruction(rhs, split_dim_rhs, split_size));

    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(dot->shape(), {split_lhs, split_rhs}));

    HloInstruction* output_tuple = splitter.BuildOutputTuple(
        -1, split_size, dot, part, /*combine_with_sum =*/true);
    HloComputation* body =
        parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));

    // build the condition
    HloComputation::Builder cond_builder("tensor_splitter_dot_cond");
    HloInstruction* cond_param =
        cond_builder.AddInstruction(HloInstruction::CreateParameter(
            0, output_tuple->shape(), "loop_param"));
    HloInstruction* cond_offset =
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_tuple->shape().tuple_shapes(0), cond_param, 0));
    HloInstruction* offset_less_than =
        cond_builder.AddInstruction(CONSTANT_INT64(main_split_size));
    HloInstruction* compare =
        cond_builder.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
            ComparisonDirection::kLt));
    HloComputation* cond =
        parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

    // build the while and replace the original element with a get
    // tuple.
    int64_t output_idx = output_tuple->shape().tuple_shapes().size() - 1;
    HloInstruction* init = dot->parent()->AddInstruction(
        HloInstruction::CreateTuple(splitter.parameters()));
    HloInstruction* loop = dot->parent()->AddInstruction(
        HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
    HloInstruction* result = dot->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(dot->shape(), loop, output_idx));

    if (split_rest == 0) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      return ReplaceInstruction(dot, result);
    } else {
      HloComputation::Builder rest_builder("tensor_splitter_dot_rest");
      Splitter splitter(rest_builder, dot->parent(),
                        absl::MakeSpan(split_leafs_lhs), main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_lhs,
          splitter.SplitInstruction(lhs, split_dim_lhs, split_rest));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_rhs,
          splitter.SplitInstruction(rhs, split_dim_rhs, split_rest));

      HloInstruction* rest_part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), {rest_lhs, rest_rhs}));

      HloComputation* rest_body =
          parent_module->AddEmbeddedComputation(rest_builder.Build(rest_part));

      splitter.parameters()[0] =
          dot->parent()->AddInstruction(CONSTANT_INT64(main_split_size));
      HloInstruction* args = dot->parent()->AddInstruction(
          HloInstruction::CreateTuple(splitter.parameters()));
      HloInstruction* rest_result = dot->parent()->AddInstruction(
          HloInstruction::CreateCall(part->shape(), {args}, rest_body));

      HloInstruction* full_result =
          dot->parent()->AddInstruction(HloInstruction::CreateBinary(
              result->shape(), HloOpcode::kAdd, result, rest_result));
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      return ReplaceInstruction(dot, full_result);
    }
  } else if ((can_split_lhs && !can_split_rhs) ||
             (!can_split_lhs && can_split_rhs)) {
    //
    // CASE :: one of lhs / rhs is split
    //
    bool split_is_lhs = can_split_lhs;
    HloInstruction* split_inst = split_is_lhs ? lhs : rhs;
    // exclude batch_dimension
    for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
      exclude_dims_lhs.push_back(dnums.lhs_batch_dimensions(i));
      exclude_dims_rhs.push_back(dnums.rhs_batch_dimensions(i));
    }
    int64_t split_dim = BestSplitDim(
        split_inst,
        absl::MakeSpan(split_is_lhs ? exclude_dims_lhs : exclude_dims_rhs));
    if (split_dim == -1) {
      // Bail, we can't split this tensor into equally sized parts.
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return Status::OK();
    }

    bool combine_parts_with_sum =
        absl::c_linear_search(split_is_lhs ? dnums.lhs_contracting_dimensions()
                                           : dnums.rhs_contracting_dimensions(),
                              split_dim);

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(split_inst, split_dim);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          split_inst->shape().dimensions(split_dim));

    ss << "\n <<< "
       << "Splitting dot '" << dot->name() << "' "
       << (split_is_lhs ? "lhs" : "rhs") << " will be split on " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    HloComputation::Builder body_builder("tensor_splitter_dot_body");
    Splitter splitter(
        body_builder, dot->parent(),
        absl::MakeSpan(split_is_lhs ? split_leafs_lhs : split_leafs_rhs));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * comp_root,
        splitter.SplitInstruction(split_inst, split_dim, split_size));

    // Add final dot inside of the computation
    HloInstruction* reduce_param;
    int64_t reduce_parameter_idx =
        splitter.AddParameter(split_is_lhs ? rhs : lhs, &reduce_param);

    Shape part_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                            dot->shape().dimensions());
    int64_t dot_split_dim = split_dim;  // split dimension after dot occured
    if (split_is_lhs) {
      for (int64_t c_dim : dnums.lhs_contracting_dimensions()) {
        if (c_dim < dot_split_dim) dot_split_dim--;
      }
    } else {
      for (int64_t c_dim : dnums.rhs_contracting_dimensions()) {
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
      int64_t param_split_dim = split_is_lhs
                                    ? dnums.rhs_contracting_dimensions()[0]
                                    : dnums.lhs_contracting_dimensions()[0];
      sliced_shape.set_dimensions(param_split_dim, split_size);

      std::vector<HloInstruction*> start_indices;
      for (int64_t dim = 0; dim < reduce_param->shape().dimensions_size();
           dim++) {
        if (dim == param_split_dim) {
          start_indices.push_back(splitter.offset());
        } else {
          start_indices.push_back(
              body_builder.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int64_t>(0))));
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
    HloComputation::Builder cond_builder("tensor_splitter_dot_cond");
    HloInstruction* cond_param =
        cond_builder.AddInstruction(HloInstruction::CreateParameter(
            0, output_tuple->shape(), "loop_param"));
    HloInstruction* cond_offset =
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_tuple->shape().tuple_shapes(0), cond_param, 0));
    HloInstruction* offset_less_than =
        cond_builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int64_t>(main_split_size)));
    HloInstruction* compare =
        cond_builder.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
            ComparisonDirection::kLt));
    HloComputation* cond =
        parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

    // build the while and replace the original element with a get
    // tuple.
    int64_t output_idx = output_tuple->shape().tuple_shapes().size() - 1;
    HloInstruction* init = dot->parent()->AddInstruction(
        HloInstruction::CreateTuple(splitter.parameters()));
    HloInstruction* loop = dot->parent()->AddInstruction(
        HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
    HloInstruction* result = dot->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(dot->shape(), loop, output_idx));

    if (split_rest == 0) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      return ReplaceInstruction(dot, result);
    } else {
      HloComputation::Builder rest_builder("tensor_splitter_dot_rest");
      Splitter splitter(
          rest_builder, dot->parent(),
          absl::MakeSpan(split_is_lhs ? split_leafs_lhs : split_leafs_rhs),
          main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * comp_root,
          splitter.SplitInstruction(split_inst, split_dim, split_rest));

      // Add final dot inside of the computation
      HloInstruction* reduce_param;
      int64_t reduce_parameter_idx =
          splitter.AddParameter(split_is_lhs ? rhs : lhs, &reduce_param);

      Shape part_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                              dot->shape().dimensions());
      int64_t dot_split_dim = split_dim;  // split dimension after dot occured
      if (split_is_lhs) {
        for (int64_t c_dim : dnums.lhs_contracting_dimensions()) {
          if (c_dim < dot_split_dim) dot_split_dim--;
        }
      } else {
        for (int64_t c_dim : dnums.rhs_contracting_dimensions()) {
          if (c_dim < dot_split_dim) dot_split_dim--;
        }
        dot_split_dim +=
            lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();
      }
      if (!combine_parts_with_sum)
        part_shape.set_dimensions(dot_split_dim, split_rest);

      if (combine_parts_with_sum) {
        Shape sliced_shape =
            ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                                 reduce_param->shape().dimensions());
        // FIXME: This assumes dots only contract once (which is currently
        // always true)
        int64_t param_split_dim = split_is_lhs
                                      ? dnums.rhs_contracting_dimensions()[0]
                                      : dnums.lhs_contracting_dimensions()[0];
        sliced_shape.set_dimensions(param_split_dim, split_rest);

        std::vector<HloInstruction*> start_indices;
        for (int64_t dim = 0; dim < reduce_param->shape().dimensions_size();
             dim++) {
          if (dim == param_split_dim) {
            start_indices.push_back(splitter.offset());
          } else {
            start_indices.push_back(
                rest_builder.AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int64_t>(0))));
          }
        }
        reduce_param =
            rest_builder.AddInstruction(HloInstruction::CreateDynamicSlice(
                sliced_shape, reduce_param, absl::MakeSpan(start_indices),
                sliced_shape.dimensions()));
      }

      std::vector<HloInstruction*> ops;
      if (split_is_lhs) {
        ops = {comp_root, reduce_param};
      } else {
        ops = {reduce_param, comp_root};
      }
      HloInstruction* part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(part_shape, absl::MakeSpan(ops)));
      HloComputation* rest_body =
          parent_module->AddEmbeddedComputation(rest_builder.Build(part));

      splitter.parameters()[0] =
          dot->parent()->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int64_t>(main_split_size)));
      HloInstruction* args = dot->parent()->AddInstruction(
          HloInstruction::CreateTuple(splitter.parameters()));
      HloInstruction* rest_result = dot->parent()->AddInstruction(
          HloInstruction::CreateCall(part->shape(), {args}, rest_body));

      if (combine_parts_with_sum) {
        HloInstruction* full_result =
            dot->parent()->AddInstruction(HloInstruction::CreateBinary(
                dot->shape(), HloOpcode::kAdd, result, rest_result));
        ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
        LOG(INFO) << ss.str();
        return ReplaceInstruction(dot, full_result);
      } else {
        Shape slice_shape = ShapeUtil::MakeShape(result->shape().element_type(),
                                                 result->shape().dimensions());
        slice_shape.set_dimensions(dot_split_dim, main_split_size);
        std::vector<int64_t> starts;
        std::vector<int64_t> limits;
        std::vector<int64_t> strides;
        for (int64_t d = 0; d < dot->shape().dimensions_size(); d++) {
          strides.push_back(1);
          starts.push_back(0);
          if (d == dot_split_dim) {
            limits.push_back(main_split_size);
          } else {
            limits.push_back(dot->shape().dimensions(d));
          }
        }
        HloInstruction* result_slice =
            dot->parent()->AddInstruction(HloInstruction::CreateSlice(
                slice_shape, result, absl::MakeSpan(starts),
                absl::MakeSpan(limits), absl::MakeSpan(strides)));
        HloInstruction* full_result =
            dot->parent()->AddInstruction(HloInstruction::CreateConcatenate(
                dot->shape(), {result_slice, rest_result}, dot_split_dim));
        ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
        LOG(INFO) << ss.str();
        return ReplaceInstruction(dot, full_result);
      }
    }
  }

  ss << "\n ----< Exit HandleDot for '" << dot->name() << "' with no splitting";
  LOG(INFO) << ss.str();
  return Status::OK();
}

Status TensorSplitterRewriteVisitor::HandleReduce(HloInstruction* reduce) {
  if (!MatchSupportedReduce(reduce)) {
    return Status::OK();
  }

  std::stringstream ss;

  LOG(INFO) << "\n =============================="
            << "\n ----> Enter HandleReduce for '" << reduce->name() << "'";

  // MatchSupportedReduce enforces that all inputs are of the
  // same shape, and that there is at least one operand!
  if (!OperandShouldBeSplit(reduce->mutable_operand(0))) {
    ss << "\n<<< Reduce '" << reduce->name()
       << "' cannot be split. Something is not splittable on the way up.";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return Status::OK();
  }

  // TODO: This is a hack, I need to more seriously rethink the
  //       two pass system, to mark elements in a first pass and combine
  //       sections properly ...
  if (OperandShouldBeSplit(reduce)) {
    ss << "\n<<< Looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return Status::OK();
  }

  // If this is a multi-argument reduce, check if only one
  // result is used.
  if (reduce->shape().IsTuple() && reduce->user_count() > 1) {
    ss << "\n<<< Nah, looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return Status::OK();
  }

  // MatchSupportedReduce enforces that all initializers are
  // scalars, so we only need to split the operands to the
  // reduce itself.
  int64_t op_count = reduce->operand_count() / 2;
  std::vector<HloInstruction*> split_leafs;
  std::vector<int64_t> orig_dims;
  std::vector<int64_t> exclude_dims;
  for (int64_t i = 0; i < op_count; i++) {
    orig_dims.clear();
    for (int64_t j = 0; j < reduce->operand(i)->shape().dimensions_size();
         j++) {
      orig_dims.push_back(j);
    }

    if (!OperandCanBeSplit(reduce->mutable_operand(i), &split_leafs, &orig_dims,
                           &exclude_dims)) {
      ss << "\n<<< Again, looks like reduce '" << reduce->name()
         << "' cannot be split because of '"
         << reduce->mutable_operand(i)->name() << "'";
      ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return Status::OK();
    }
  }

  if (reduce->shape().IsTuple()) {
    for (int64_t reduce_dim : reduce->dimensions()) {
      exclude_dims.push_back(reduce_dim);
    }
  }

  int64_t split_dim =
      BestSplitDim(reduce->mutable_operand(0), absl::MakeSpan(exclude_dims));
  if (split_dim == -1) {
    // Bail, we can't split this tensor into equally sized parts.
    ss << "\n<<< Looks like reduce '" << reduce->name()
       << "' cannot be split into equally sized parts";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return Status::OK();
  }

  bool split_along_reduce_dim =
      absl::c_linear_search(reduce->dimensions(), split_dim);

  int64_t split_size, split_count, split_rest;
  std::tie(split_size, split_count, split_rest) =
      DetermineSplitSize(reduce->mutable_operand(0), split_dim);

  auto main_split_size = split_count * split_size;
  CHECK(main_split_size + split_rest ==
        reduce->mutable_operand(0)->shape().dimensions(split_dim));

  ss << "<<< "
     << "Splitting reduce " << reduce->name()
     << " operands will be split at dimension " << split_dim
     << " with split size " << split_size << " and rest size " << split_rest;

  HloComputation::Builder body_builder("tensor_splitter_reduce_body");
  Splitter splitter(body_builder, reduce->parent(),
                    absl::MakeSpan(split_leafs));

  std::vector<HloInstruction*> operands;
  for (int64_t i = 0; i < op_count; i++) {
    TF_ASSIGN_OR_RETURN(HloInstruction * split_op,
                        splitter.SplitInstruction(reduce->mutable_operand(i),
                                                  split_dim, split_size));
    operands.push_back(split_op);
  }

  // Add init parameters to computation
  for (int64_t i = 0; i < op_count; i++) {
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

  int64_t reduce_split_dim = split_dim;  // split dim after reduce
  for (int64_t r_dim : reduce->dimensions()) {
    if (r_dim < split_dim) {
      reduce_split_dim--;
    }
  }

  Shape output_part_shape;
  HloInstruction *output_part, *old_output;
  if (reduce->shape().IsTuple()) {
    CHECK(reduce->user_count() == 1);
    old_output = reduce->users()[0];

    Shape new_reduce_shape = ShapeUtil::MakeTupleShape(
        absl::MakeSpan(reduce->shape().tuple_shapes()));
    if (!split_along_reduce_dim) {
      for (int64_t i = 0; i < new_reduce_shape.tuple_shapes_size(); i++) {
        new_reduce_shape.mutable_tuple_shapes(i)->set_dimensions(
            reduce_split_dim, split_size);
      }
    }
    HloInstruction* new_reduce = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(new_reduce_shape, operands));

    output_part_shape = ShapeUtil::MakeShape(old_output->shape().element_type(),
                                             old_output->shape().dimensions());
    if (!split_along_reduce_dim) {
      output_part_shape.set_dimensions(reduce_split_dim, split_size);
    }
    output_part =
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_part_shape, new_reduce, old_output->tuple_index()));
  } else {
    output_part_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                             reduce->shape().dimensions());
    if (!split_along_reduce_dim)
      output_part_shape.set_dimensions(reduce_split_dim, split_size);
    output_part = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(output_part_shape, operands));
    old_output = reduce;
  }

  HloInstruction* output_tuple =
      splitter.BuildOutputTuple(reduce_split_dim, split_size, old_output,
                                output_part, false, split_along_reduce_dim);
  HloComputation* body =
      parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));

  // build the condition
  HloComputation::Builder cond_builder("tensor_splitter_reduce_cond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, output_tuple->shape(), "loop_param"));
  HloInstruction* cond_offset =
      cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          output_tuple->shape().tuple_shapes(0), cond_param, 0));
  HloInstruction* offset_less_than =
      cond_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int64_t>(main_split_size)));
  HloInstruction* compare =
      cond_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
          ComparisonDirection::kLt));
  HloComputation* cond =
      parent_module->AddEmbeddedComputation(cond_builder.Build(compare));

  // build the while and replace the original element with a get
  // tuple.
  int64_t output_idx = output_tuple->shape().tuple_shapes().size() - 1;
  HloInstruction* init = reduce->parent()->AddInstruction(
      HloInstruction::CreateTuple(splitter.parameters()));
  HloInstruction* loop = reduce->parent()->AddInstruction(
      HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));
  HloInstruction* result =
      reduce->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
          old_output->shape(), loop, output_idx));

  if (split_rest > 0) {
    HloComputation::Builder rest_builder("tensor_splitter_reduce_rest");
    Splitter rest_splitter(rest_builder, reduce->parent(),
                           absl::MakeSpan(split_leafs), main_split_size);

    std::vector<HloInstruction*> operands;
    for (int64_t i = 0; i < op_count; i++) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * split_op,
          rest_splitter.SplitInstruction(reduce->mutable_operand(i), split_dim,
                                         split_rest));
      operands.push_back(split_op);
    }

    // Add init parameters to computation
    for (int64_t i = 0; i < op_count; i++) {
      HloInstruction* init_op;
      rest_splitter.AddParameter(reduce->mutable_operand(i + op_count),
                                 &init_op);
      operands.push_back(init_op);
    }

    int64_t reduce_split_dim = split_dim;
    for (int64_t r_dim : reduce->dimensions()) {
      if (r_dim < split_dim) {
        reduce_split_dim--;
      }
    }

    Shape output_part_shape;
    HloInstruction *output_part, *old_output;
    if (reduce->shape().IsTuple()) {
      CHECK(reduce->user_count() == 1);
      old_output = reduce->users()[0];

      Shape new_reduce_shape = ShapeUtil::MakeTupleShape(
          absl::MakeSpan(reduce->shape().tuple_shapes()));
      if (!split_along_reduce_dim) {
        for (int64_t i = 0; i < new_reduce_shape.tuple_shapes_size(); i++) {
          new_reduce_shape.mutable_tuple_shapes(i)->set_dimensions(
              reduce_split_dim, split_rest);
        }
      }
      HloInstruction* new_reduce = rest_builder.AddInstruction(
          reduce->CloneWithNewOperands(new_reduce_shape, operands));

      output_part_shape = ShapeUtil::MakeShape(
          old_output->shape().element_type(), old_output->shape().dimensions());
      if (!split_along_reduce_dim) {
        output_part_shape.set_dimensions(reduce_split_dim, split_rest);
      }
      output_part =
          rest_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              output_part_shape, new_reduce, old_output->tuple_index()));
    } else {
      output_part_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                               reduce->shape().dimensions());
      if (!split_along_reduce_dim) {
        output_part_shape.set_dimensions(reduce_split_dim, split_rest);
      }
      output_part = rest_builder.AddInstruction(
          reduce->CloneWithNewOperands(output_part_shape, operands));
      old_output = reduce;
    }

    HloInstruction* output_tuple = rest_splitter.BuildOutputTuple(
        reduce_split_dim, split_rest, old_output, output_part, false,
        split_along_reduce_dim);
    HloComputation* rest =
        parent_module->AddEmbeddedComputation(rest_builder.Build(output_tuple));

    int64_t output_idx = output_tuple->shape().tuple_shapes().size() - 1;
    HloInstruction* init = reduce->parent()->AddInstruction(
        HloInstruction::CreateTuple(rest_splitter.parameters()));
    HloInstruction* call = reduce->parent()->AddInstruction(
        HloInstruction::CreateCall(output_tuple->shape(), {init}, rest));
    HloInstruction* result_rest =
        reduce->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            old_output->shape(), call, output_idx));

    if (split_along_reduce_dim) {
      // we're splitting on a reduced dimension
      CHECK(reduce->opcode() == HloOpcode::kReduce);
      CHECK(reduce->operand_count() == 2);
      HloComputation* reduce_fn = reduce->called_computations()[0];
      if (ShapeUtil::IsScalar(result_rest->shape())) {
        // we can call the function directly
        result = reduce->parent()->AddInstruction(HloInstruction::CreateCall(
            reduce->shape(), {result, result_rest}, reduce_fn));
      } else {
        // we have to call the function through map
        result = reduce->parent()->AddInstruction(HloInstruction::CreateMap(
            reduce->shape(), {result, result_rest}, reduce_fn));
      }
    } else {
      auto dim_len = old_output->shape().dimensions_size();
      Shape slice_shape = ShapeUtil::MakeShape(result->shape().element_type(),
                                               result->shape().dimensions());
      slice_shape.set_dimensions(reduce_split_dim, main_split_size);
      std::vector<int64_t> starts;
      std::vector<int64_t> limits;
      std::vector<int64_t> strides;

      for (int64_t d = 0; d < dim_len; d++) {
        strides.push_back(1);
        starts.push_back(0);
        if (d == reduce_split_dim) {
          limits.push_back(main_split_size);
        } else {
          limits.push_back(old_output->shape().dimensions(d));
        }
      }

      HloInstruction* result_slice =
          reduce->parent()->AddInstruction(HloInstruction::CreateSlice(
              slice_shape, result, absl::MakeSpan(starts),
              absl::MakeSpan(limits), absl::MakeSpan(strides)));

      Shape slice_shape_rest =
          ShapeUtil::MakeShape(result_rest->shape().element_type(),
                               result_rest->shape().dimensions());
      slice_shape_rest.set_dimensions(reduce_split_dim, split_rest);
      std::vector<int64_t> starts_rest;
      std::vector<int64_t> limits_rest;
      std::vector<int64_t> strides_rest;

      for (int64_t d = 0; d < dim_len; d++) {
        strides_rest.push_back(1);
        auto full_size = old_output->shape().dimensions(d);
        limits_rest.push_back(full_size);
        if (d == reduce_split_dim) {
          starts_rest.push_back(main_split_size);
        } else {
          starts_rest.push_back(0);
        }
      }

      HloInstruction* result_rest_slice =
          reduce->parent()->AddInstruction(HloInstruction::CreateSlice(
              slice_shape_rest, result_rest, absl::MakeSpan(starts_rest),
              absl::MakeSpan(limits_rest), absl::MakeSpan(strides_rest)));

      // LOG(INFO) << "# Shape of the main slice: " << result_slice->shape();
      // LOG(INFO) << "# Shape of the rest slice: " <<
      // result_rest_slice->shape(); LOG(INFO) << "# Exit via splitting on main
      // and rest";

      result =
          reduce->parent()->AddInstruction(HloInstruction::CreateConcatenate(
              reduce->shape(), {result_slice, result_rest_slice},
              reduce_split_dim));
    }
  }
  ss << "\n <---- Exit HandleReduce for '" << old_output->name() << "' SUCCESS";
  LOG(INFO) << ss.str();
  return ReplaceInstruction(old_output, result);
}

Status TensorSplitterRewriteVisitor::HandleSort(HloInstruction* sort) {
  CHECK(Match(sort, m::Sort()));
  if (!Match(sort, m::Sort(m::Op(), m::Iota()))) {
    LOG(WARNING) << "Splitter cannot transform '" << sort->name()
                 << "' which is not supported sort-type operation. "
                 << "Iota is expected as the second parameter";
    return Status::OK();
  }
  HloInstruction* array = sort->mutable_operand(0);
  HloInstruction* indices = sort->mutable_operand(1);

  std::stringstream msg;

  if (!ShapeUtil::CompatibleIgnoringElementType(array->shape(),
                                                indices->shape())) {
    return Status::OK();
  }

  HloInstruction* get_array;
  HloInstruction* get_indices;
  if (sort->shape().IsTuple() && sort->shape().tuple_shapes_size() == 2 &&
      sort->user_count() == 2) {
    get_array = sort->users()[0];
    get_indices = sort->users()[1];
    if (get_array->user_count() != 1 || get_indices->user_count() != 1) {
      LOG(WARNING) << "Splitting pattern doesn't match sort operation as "
                   << "number of users on the left is "
                   << get_array->user_count() << " and on the right is "
                   << get_indices->user_count();
      return Status::OK();
    }
  }

  HloInstruction* array_slice = get_array->users()[0];
  HloInstruction* indices_slice = get_indices->users()[0];
  auto left_is_slice =
      Match(array_slice, m::Slice(m::GetTupleElement(m::Sort())));
  auto right_is_slice =
      Match(indices_slice, m::Slice(m::GetTupleElement(m::Sort())));
  if (!(left_is_slice && right_is_slice)) {
    return Status::OK();
  }

  if (!ShapeUtil::CompatibleIgnoringElementType(array_slice->shape(),
                                                indices_slice->shape())) {
    return Status::OK();
  }

  // Checks that the operation can be split
  auto array_dims_size = array->shape().dimensions().size();
  std::vector<int64_t> original_dims(array_dims_size);
  std::iota(original_dims.begin(), original_dims.end(), 0);

  std::vector<int64_t> exclude_dims;
  std::vector<HloInstruction*> split_leafs;
  std::vector<HloInstruction*> indices_split_leafs;

  bool can_split_array =
      OperandShouldBeSplit(array) &&
      OperandCanBeSplit(array, &split_leafs, &original_dims, &exclude_dims);

  bool can_split_indices = OperandCanBeSplit(indices, &indices_split_leafs,
                                             &original_dims, &exclude_dims);

  if (!(can_split_array && can_split_indices)) {
    LOG(WARNING) << "Operation '" << sort->name()
                 << "' either does not require splitting "
                 << "or cannot be splitted";
    return Status::OK();
  }

  auto split_dim = BestSplitDim(array, absl::MakeSpan(exclude_dims));
  if (split_dim == -1 || absl::c_linear_search(exclude_dims, split_dim)) {
    LOG(WARNING) << "Failed to find best split dimension for '" << sort->name()
                 << "'";
    return Status::OK();
  }

  auto last_dim = array_dims_size - 1;
  if (split_dim != last_dim) {
    LOG(WARNING) << "Best split dimension for '" << sort->name()
                 << "' can only be the last dimension. "
                 << "Best found split dimension is " << split_dim;
    return Status::OK();
  }

  int64_t split_size, split_count, split_rest;
  std::tie(split_size, split_count, split_rest) =
      DetermineSplitSize(array, split_dim);

  auto slice_k = array_slice->shape().dimensions(last_dim);
  if (slice_k >= split_size) {
    LOG(WARNING) << "Splitting for '" << sort->name()
                 << "' will not benefit user as the slicing dimension ("
                 << slice_k << ") "
                 << "is larger or equal to the split size (" << split_size
                 << ")";
    return Status::OK();
  }

  auto main_split_size = split_count * split_size;
  CHECK(main_split_size + split_rest == array->shape().dimensions(split_dim));

  HloComputation* sort_comp = sort->parent();

  // msg << "\n -> Split dim=" << split_dim << ", size=" << split_size
  //     << ", count=" << split_count << ", rest=" << split_rest
  //     << ", slice_k=" << slice_k;

  // Build While body

  absl::c_move(indices_split_leafs, std::back_inserter(split_leafs));

  auto builder_name = "splitter_body_sort";
  HloComputation::Builder body_builder(builder_name);
  Splitter body_splitter(body_builder, sort_comp, absl::MakeSpan(split_leafs));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * body_array,
      body_splitter.SplitInstruction(array, split_dim, split_size));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * body_indices,
      body_splitter.SplitInstruction(indices, split_dim, split_size));

  HloInstruction* body_offset = body_splitter.offset();
  HloInstruction* body_split_size =
      body_builder.AddInstruction(CONSTANT_INT64(split_size));

  // Create the body of while loop
  // The body sort operation acts on slices of the original tensor
  Shape body_sort_shape =
      ShapeUtil::MakeTupleShape({body_array->shape(), body_indices->shape()});
  HloInstruction* body_sort = body_builder.AddInstruction(
      sort->CloneWithNewOperands(body_sort_shape, {body_array, body_indices}));
  HloInstruction* body_get_array =
      body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          body_sort_shape.tuple_shapes(0), body_sort, 0));
  HloInstruction* body_get_indices =
      body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          body_sort_shape.tuple_shapes(1), body_sort, 1));

  HloInstruction* body_array_slice =
      body_builder.AddInstruction(array_slice->CloneWithNewOperands(
          array_slice->shape(), {body_get_array}));
  HloInstruction* body_indices_slice =
      body_builder.AddInstruction(indices_slice->CloneWithNewOperands(
          indices_slice->shape(), {body_get_indices}));

  // Initialize While sort input from outside of the loop
  // and place initial arrays tuple.
  // auto split_dim_size = slice_k * split_count; // TODO

  Shape body_acc_array_shape = array_slice->shape();
  Shape body_acc_indices_shape = indices_slice->shape();
  // body_acc_array_shape.set_dimensions(split_dim, split_dim_size); // TODO
  // body_acc_indices_shape.set_dimensions(split_dim, split_dim_size); // TODO
  Shape body_acc_shape =
      ShapeUtil::MakeTupleShape({body_acc_array_shape, body_acc_indices_shape});

  HloInstruction* body_init_acc_array_scalar =
      sort_comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(body_acc_shape.tuple_shapes(0).element_type())));
  HloInstruction* body_init_acc_indices_scalar =
      sort_comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(body_acc_shape.tuple_shapes(1).element_type())));

  HloInstruction* body_init_acc_array =
      sort_comp->AddInstruction(HloInstruction::CreateBroadcast(
          body_acc_shape.tuple_shapes(0), body_init_acc_array_scalar, {}));
  HloInstruction* body_init_acc_indices =
      sort_comp->AddInstruction(HloInstruction::CreateBroadcast(
          body_acc_shape.tuple_shapes(1), body_init_acc_indices_scalar, {}));

  HloInstruction* body_init_acc =
      sort_comp->AddInstruction(HloInstruction::CreateTuple(
          {body_init_acc_array, body_init_acc_indices}));

  HloInstruction* body_get_acc;
  auto body_acc_parameter_id =
      body_splitter.AddParameter(body_init_acc, &body_get_acc);

  // While operation updates
  // Update While accumulator for sort operation.

  // Continue building body. Sorting cached and new arrays.
  //

  HloInstruction* body_cond_params =
      body_builder.AddInstruction(HloInstruction::CreateTuple(
          {body_array_slice, body_indices_slice, body_get_acc}));

  const Shape& body_cond_params_shape = body_cond_params->shape();

  HloComputation::Builder false_builder("splitter_body_false_cond");
  {
    auto false_param =
        false_builder.AddInstruction(HloInstruction::CreateParameter(
            0, body_cond_params_shape, "body_sort_result"));

    auto false_array_slice_new =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(0), false_param, 0));
    auto false_indices_slice_new =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(1), false_param, 1));

    const Shape& false_acc_shape = body_cond_params_shape.tuple_shapes(2);
    auto false_acc = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(false_acc_shape, false_param, 2));

    auto false_array_acc =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_acc_shape.tuple_shapes(0), false_acc, 0));
    auto false_indices_acc =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_acc_shape.tuple_shapes(1), false_acc, 1));

    Shape false_array_shape = body_acc_array_shape;
    Shape false_indices_shape = body_acc_indices_shape;
    false_array_shape.set_dimensions(
        split_dim, 2 * body_acc_array_shape.dimensions(split_dim));
    false_indices_shape.set_dimensions(
        split_dim, 2 * body_acc_indices_shape.dimensions(split_dim));
    Shape false_sort_shape =
        ShapeUtil::MakeTupleShape({false_array_shape, false_indices_shape});

    auto false_array =
        false_builder.AddInstruction(HloInstruction::CreateConcatenate(
            false_sort_shape.tuple_shapes(0),
            {false_array_slice_new, false_array_acc}, split_dim));
    auto false_indices =
        false_builder.AddInstruction(HloInstruction::CreateConcatenate(
            false_sort_shape.tuple_shapes(1),
            {false_indices_slice_new, false_indices_acc}, split_dim));

    auto false_sort = false_builder.AddInstruction(sort->CloneWithNewOperands(
        false_sort_shape, {false_array, false_indices}));

    auto false_array_get =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_sort_shape.tuple_shapes(0), false_sort, 0));
    auto false_indices_get =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_sort_shape.tuple_shapes(1), false_sort, 1));

    auto false_array_slice =
        false_builder.AddInstruction(array_slice->CloneWithNewOperands(
            array_slice->shape(), {false_array_get}));
    auto false_indices_slice =
        false_builder.AddInstruction(indices_slice->CloneWithNewOperands(
            indices_slice->shape(), {false_indices_get}));

    false_builder.AddInstruction(
        HloInstruction::CreateTuple({false_array_slice, false_indices_slice}));
  }
  HloComputation* false_comp =
      parent_module->AddEmbeddedComputation(false_builder.Build());

  HloComputation::Builder true_builder("splitter_body_true_cond");
  {
    auto true_param =
        true_builder.AddInstruction(HloInstruction::CreateParameter(
            0, body_cond_params_shape, "body_sort_result"));

    auto true_array =
        true_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(0), true_param, 0));
    auto true_indices =
        true_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(1), true_param, 1));

    true_builder.AddInstruction(
        HloInstruction::CreateTuple({true_array, true_indices}));
  }
  HloComputation* true_comp =
      parent_module->AddEmbeddedComputation(true_builder.Build());

  HloInstruction* body_init_offset =
      body_builder.AddInstruction(CONSTANT_INT64(0));
  HloInstruction* body_cond_pred =
      body_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), body_offset, body_init_offset,
          ComparisonDirection::kEq));

  HloInstruction* body_update_acc =
      body_builder.AddInstruction(HloInstruction::CreateConditional(
          body_acc_shape, body_cond_pred, body_cond_params, true_comp,
          body_cond_params, false_comp));

  // Update split slice iteration
  HloInstruction* body_update_offset =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          body_offset->shape(), HloOpcode::kAdd, body_offset, body_split_size));

  // Collect all updates to the input parameters into a tuple.
  HloInstruction* while_parameters = body_splitter.tuple_parameters();
  const Shape& while_parameters_shape = while_parameters->shape();

  std::vector<HloInstruction*> while_parameter_updates = {body_update_offset};
  for (auto id = 1; id < while_parameters_shape.tuple_shapes_size(); id++) {
    if (id == body_acc_parameter_id) {
      while_parameter_updates.push_back(body_update_acc);
    } else {
      HloInstruction* body_get_input =
          body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              while_parameters_shape.tuple_shapes(id), while_parameters, id));
      while_parameter_updates.push_back(body_get_input);
    }
  }

  HloInstruction* body_parameter_updates = body_builder.AddInstruction(
      HloInstruction::CreateTuple(while_parameter_updates));
  HloComputation* body = parent_module->AddEmbeddedComputation(
      body_builder.Build(body_parameter_updates));

  // Create While loop, including condition for While loop.
  auto parameters_shape = body_parameter_updates->shape();
  HloComputation* condition = CreateWhileSplitCondition(
      "splitter_condition_sort", parameters_shape, main_split_size);
  auto results = CreateWhileSplitWithResults(
      sort_comp, condition, body, body_splitter.parameters(),
      {std::make_tuple(body_acc_parameter_id, body_acc_shape)});
  HloInstruction* while_sort_result = results[0];

  HloInstruction* while_get_array =
      sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          body_acc_shape.tuple_shapes(0), while_sort_result, 0));
  HloInstruction* while_get_indices =
      sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          body_acc_shape.tuple_shapes(1), while_sort_result, 1));

  CHECK(while_get_array->shape().dimensions(split_dim) == slice_k);
  CHECK(while_get_indices->shape().dimensions(split_dim) == slice_k);

  // Build
  if (split_rest == 0) {
    TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
        array_slice, while_get_array));
    TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
        indices_slice, while_get_indices));
  } else {
    auto rest_builder_name = "splitter_rest_sort";
    HloComputation::Builder rest_builder(rest_builder_name);
    Splitter rest_splitter(rest_builder, sort_comp, absl::MakeSpan(split_leafs),
                           main_split_size);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * rest_array,
        rest_splitter.SplitInstruction(array, split_dim, split_rest));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * rest_indices,
        rest_splitter.SplitInstruction(indices, split_dim, split_rest));

    HloInstruction* get_params_array = nullptr;
    HloInstruction* get_params_indices = nullptr;
    rest_splitter.AddParameter(while_get_array, &get_params_array);
    rest_splitter.AddParameter(while_get_indices, &get_params_indices);

    Shape merged_array_shape = while_get_array->shape();
    Shape merged_indices_shape = while_get_indices->shape();
    auto merged_split_dim_size = slice_k + split_rest;
    merged_array_shape.set_dimensions(split_dim, merged_split_dim_size);
    merged_indices_shape.set_dimensions(split_dim, merged_split_dim_size);
    auto rest_merged_array =
        rest_builder.AddInstruction(HloInstruction::CreateConcatenate(
            merged_array_shape, {get_params_array, rest_array}, split_dim));
    auto rest_merged_indices =
        rest_builder.AddInstruction(HloInstruction::CreateConcatenate(
            merged_indices_shape, {get_params_indices, rest_indices},
            split_dim));

    Shape rest_sort_shape = ShapeUtil::MakeTupleShape(
        {rest_merged_array->shape(), rest_merged_indices->shape()});
    auto rest_sort = rest_builder.AddInstruction(sort->CloneWithNewOperands(
        rest_sort_shape, {rest_merged_array, rest_merged_indices}));

    HloComputation* rest_comp =
        parent_module->AddEmbeddedComputation(rest_builder.Build(rest_sort));

    rest_splitter.parameters()[0] =
        sort_comp->AddInstruction(CONSTANT_INT64(main_split_size));
    auto rest_args = sort_comp->AddInstruction(
        HloInstruction::CreateTuple(rest_splitter.parameters()));
    auto rest_sort_result = sort_comp->AddInstruction(
        HloInstruction::CreateCall(rest_sort_shape, {rest_args}, rest_comp));

    auto rest_get_array =
        sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            rest_sort_shape.tuple_shapes(0), rest_sort_result, 0));
    auto rest_get_indices =
        sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            rest_sort_shape.tuple_shapes(1), rest_sort_result, 1));

    auto rest_array_slice =
        sort_comp->AddInstruction(array_slice->CloneWithNewOperands(
            array_slice->shape(), {rest_get_array}));
    auto rest_indices_slice =
        sort_comp->AddInstruction(indices_slice->CloneWithNewOperands(
            indices_slice->shape(), {rest_get_indices}));

    TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
        array_slice, rest_array_slice));
    TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
        indices_slice, rest_indices_slice));
  }

  changed_ = true;
  return Status::OK();
}

bool endsWith(const std::string& str, std::string pattern) {
  if (pattern.size() > str.size()) return false;
  for (int i = 1; i <= pattern.size(); i++) {
    if (pattern[pattern.size() - i] != str[str.size() - i]) return false;
  }
  return true;
}

std::tuple<int64_t, int64_t> TensorSplitter::SplitSettings() {
  auto tensor_size_threshold =
      GetDebugOptionsFromFlags().xla_tensor_size_threshold();
  auto tensor_split_size = GetDebugOptionsFromFlags().xla_tensor_split_size();

  auto size_threshold = TensorBytes(tensor_size_threshold);
  auto split_size = TensorBytes(tensor_split_size);
  if (split_size == 0) {
    split_size = size_threshold;
  }
  return std::make_tuple(size_threshold, split_size);
}

int64_t TensorSplitter::TensorBytes(const std::string& option) {
  int64_t raw = (int64_t)atoi(option.c_str());
  if (raw <= 0) {
    return 134217728;  // 1 GiB
  }

  if (endsWith(option, "GB") || endsWith(option, "gb"))
    return raw * 1000000000;  // 1e9
  else if (endsWith(option, "GiB"))
    return raw * 134217728;
  else if (endsWith(option, "MB") || endsWith(option, "mb"))
    return raw * 1000000;  // 1e6
  else if (endsWith(option, "MiB"))
    return raw * 1048576;
  else if (endsWith(option, "kB") || endsWith(option, "kb"))
    return raw * 1000;
  else if (endsWith(option, "kiB"))
    return raw * 1024;
  else
    return raw;  // interpret as bytes
}

StatusOr<bool> TensorSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int64_t size_threshold;
  int64_t split_size;
  std::tie(size_threshold, split_size) = SplitSettings();
  TensorSplitterRewriteVisitor rewrite(size_threshold, split_size, module);
  LOG(INFO) << "Running tensor splitter for '" << module->name() << "'";
  return rewrite.RunOnModule(module, execution_threads);
}

}  // namespace xla
