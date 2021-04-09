
#include "tensorflow/compiler/xla/service/do_random_shit.h"

#include <stdio.h>

namespace xla {

StatusOr<bool> RandomShitPass::Run(HloModule* module) {
  if (!module->has_entry_computation()) {
    printf("DELETE ME: no entry computatio\n");
    return false;
  }

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();

  if (!root) {
    printf("DELETE ME: No root instruction\n");
    return false;
  }

  printf("Root OptCode: %s\n", HloOpcodeString(root->opcode()).c_str());

  return false;
}

} // namespace xla
