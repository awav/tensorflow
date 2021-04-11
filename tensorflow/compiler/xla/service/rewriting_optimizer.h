// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_REWRITING_OPTIMIZER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_REWRITING_OPTIMIZER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which searches equivalent rewrites to reduce memory-consuption of
// intermediate results and reduce overall computation time.
class RewritingOptimizer : public HloModulePass {
  public:
    absl::string_view name() const override { return "rewriting-optimizer"; }
  
  // Searches and applies equivalent rewrites to the graph
  StatusOr<bool> Run(HloModule* module) override;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_REWRITING_OPTIMIZER_
