// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALGERAIC_REWRITER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALGERAIC_REWRITER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which applies algebraic equivalences
class AlgebraicRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "algebraic-rewriter"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGERAIC_REWRITER_
