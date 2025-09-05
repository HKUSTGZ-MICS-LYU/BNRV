# RISC-V BitNet Project
Based on `NaxRiscvSoftware` but self-contained. Supposed to be run on the VexiiRiscv CPU and its infrastructure.

Requires RISC-V GNU toolchain.

Custom instructions for BitNet SIMD included.

## Make Arguments

| Arg | Options | Description |
| --- | ------- | ----------- |
| `MARCH` | `rv32ic`, `rv32imc`, `rv32imfc` | Target ISA |
| `USE_SIMD` | `0`, `4`, `8`, `16`, `32`, `64` | `0` - No Custom SIMD, others - Level of SIMD |
| `BITNET_QTYPE` | `2`, `3`, `4` | BitNet Quantization Type (1, 1.58, 2-bit) |

## RISC-V BitNet SIMD ISA Extension Spec.

We extends RISC-V **32-bit** ISA for efficient BitNet **multiplication-free** inference computation.

### Without BitNet Weight Buffer:
For 4x8-bit SIMD, we use a single buffer-free SIMD instruction:
| Instruction | Syntax |
| ----------- | ------ |
| `BN.SUM4`     | `BN.SUM4 rd (32-bit), rs1 (4*8-bit), rs2 (4*1-bit / 4*2-bit)`|

**Description:** 

Compute the sum of four 8-bit signed integer inputs according to four 1-bit or 2-bit BitNet quantized weights. The `rs1` should contain a 32-bit packed 4*8-bit vector, the lower part of `rs` (i.e. `rs2[3:0]` or `rs2[7:0]`) should contain 4 BitNet quantized weights. The inputs are toggled, flipped or shifted according to the weights, then summed up to get the result. The 32-bit computation result is written to `rd`.

**Note:** Although the instruction is quite simple and efficient, it suffers from low bit utilization, as only the lower part of the `rs2` is used. This may increase the memory footprint.

### With BitNet Weight Buffer:
For 8x8-bit SIMD, we use a store instruction the weights into a weight buffer first, then use a SIMD instruction to sum the inputs:

Denote `N` as the number of weights that can be stored in the buffer (buffer width), then `M=M1+M2=N*Q`, where `Q` is the bit width of the weights (1 or 2 for BitNet). `N` should be larger than 8.

| Instruction | Syntax |
| ----------- | ------ |
| `BN.STORE`  | `BN.STORE x0, rs1 (M2-bit), rs2 (M1-bit)` |

**Description:** 

Store the concatentated BitNet weights `rs1 ## rs2` into the weight buffer for following computations. The stored part is `(rs1 ## rs2)[N*Q-1:0]`, which means the upper part will be ignored if `N*Q` < 64. When `N` > 8, there will be a offset pointer that points to current group of weights.

**Note**: The instruction doesn't write back anything, so the `rd` is fixed to `x0`.

| Instruction | Syntax |
| ----------- | ------ |
| `BN.SUM8`   | `BN.SUM8 rd (32-bit), rs1 (4*8-bit), rs2 (4*8-bit)`|

Compute the sum of eight 8-bit signed integer inputs according to eight 1-bit or 2-bit BitNet quantized weights stored in the weight buffer. Both the `rs1` and `rs2` should contain a 32-bit packed 4*8-bit vector. The inputs are toggled, flipped or shifted according to the weights, then summed up to get the result. The 32-bit computation result is written to `rd`. 

**Note**: When `N` > 8, after each execution of this instruction, the offset pointer will be incremented to point at the next group of weights.

For usage examples, please check `bitnet.S` and `bitnet.h`.