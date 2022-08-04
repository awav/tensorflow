// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which splits tensor values
class TensorSplitter : public HloModulePass {
 public:
  absl::string_view name() const override { return "tensor-splitter"; }

  StatusOr<bool> Run(HloModule* module) override;

  // Use this to retreive the configured split size in bytes.
  static int64_t TensorBytes(const std::string& option);
  static std::tuple<int64_t, int64_t> SplitSettings();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_
