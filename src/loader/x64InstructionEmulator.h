#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <Zydis/DecoderTypes.h>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                                         const ZydisDecodedOperand* operands);
uint64_t           PatchReciprocalSquareRoots(uint64_t address, uint64_t size);
[[nodiscard]] bool TryEmulate(void* native_context);

// Reads [vaddr, vaddr + size) for a load that faulted at fault_vaddr; false to decline.
using LoadReader = bool (*)(uint64_t fault_vaddr, uint64_t vaddr, void* data, uint64_t size);
// Completes a faulting load whose memory operand covers fault_vaddr by reading the operand through
// `read` instead of the faulting mapping, then steps past the instruction. Pure loads only: MOV,
// MOVZX, MOVSX, MOVSXD, CMP and TEST with a memory operand, and SSE/AVX moves from memory into a
// vector register. False, with the context untouched, for anything else or when `read` declines.
[[nodiscard]] bool TryEmulateLoad(void* native_context, uint64_t fault_vaddr, LoadReader read);

// Writes [vaddr, vaddr + size) for a store that faulted at fault_vaddr; false to decline.
using StoreWriter = bool (*)(uint64_t fault_vaddr, uint64_t vaddr, const void* data, uint64_t size);
// Completes a faulting store whose memory operand covers fault_vaddr by handing the stored bytes
// to `write` instead of the faulting mapping, then steps past the instruction. Pure stores only
// (the old value is never needed): MOV and MOVNTI from a register or immediate, and SSE/AVX moves
// from a vector register to memory. False, with the context untouched, for anything else (LOCK,
// read-modify-write, string instructions) or when `write` declines.
[[nodiscard]] bool TryEmulateStore(void* native_context, uint64_t fault_vaddr, StoreWriter write);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
