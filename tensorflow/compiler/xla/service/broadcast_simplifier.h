// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BROADCAST_SIMPLIFIER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BROADCAST_SIMPLIFIER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which removes unnecessary broadcasts. Mainly meant to
// aide passes like intermediate_tensor_splitter.
class BroadcastSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override { return "split-intermediate-tensors"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BROADCAST_SIMPLIFIER_
