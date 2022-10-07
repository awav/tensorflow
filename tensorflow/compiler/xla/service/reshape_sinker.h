// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// The role of reshape-sinker is to sink the reshape operations in a
// computational graph below dot operations, allowing the tensor-splitter to
// split more subgraphs.
// For example:
//  [2,3,5]           [5,4]               [2,3,5]      [5,4]
//     |                /                     \         /
//   reshape(6,5)      /          ->           dot([2,3,5])
//          \         /                             |
//            dot([6,5])                        reshape([6,5])

class ReshapeSinker : public HloModulePass {
 public:
  absl::string_view name() const override { return "reshape-sinker"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_
