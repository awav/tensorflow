
#include "tensorflow/compiler/xla/service/do_random_shit.h"

#include <stdio.h>

namespace xla {

StatusOr<bool> RandomShitPass::Run(HloModule* module) {
  printf("This is a test!\n");
  return false;
}

} // namespace xla
