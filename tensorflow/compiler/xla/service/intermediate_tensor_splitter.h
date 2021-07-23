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

  StatusOr<bool> Run(HloModule* module) override;

  // Use this to retreive the configured split size in bytes.
  static int64 SplitTensorBytes();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERMEDIATE_TENSOR_SPLITTER_
