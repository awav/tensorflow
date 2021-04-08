
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DO_RANDOM_SHIT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DO_RANDOM_SHIT_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which does some random shit.
class RandomShitPass : public HloModulePass {
 public:
  RandomShitPass() {}
  ~RandomShitPass() override = default;
  absl::string_view name() const override { return "algrandshit"; }

  // Does some random things with the graph
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DO_RANDOM_SHIT_H_
