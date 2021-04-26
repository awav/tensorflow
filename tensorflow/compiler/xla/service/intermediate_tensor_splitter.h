// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERMEDIATE_TENSOR_SPLITTER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERMEDIATE_TENSOR_SPLITTER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which splits intermediate tensor values
class IntermediateTensorSplitter : public HloModulePass {
 public:
  absl::string_view name() const override { return "split-intermediate-tensors"; }

  // Searches for newsted dots and reorders them
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERMEDIATE_TENSOR_SPLITTER_
