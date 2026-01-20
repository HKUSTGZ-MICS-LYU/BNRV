# BNRV Custom Instructions Quick Reference

This document provides a concise reference for the BNRV custom RISC-V instructions.

## Instruction Summary

| Instruction | Opcode | func3 | func7 | Description |
|------------|--------|-------|-------|-------------|
| BNSTORE    | 0x0B   | 0x2   | 0x00  | Store packed weights to hardware buffer |
| BNSUM      | 0x0B   | 0x1   | 0x00  | BitNet multiply-accumulate |
| DOTP8      | 0x0B   | 0x4   | 0x01  | 8-way dot product |
| DOTP8X2    | 0x0B   | 0x5   | 0x01  | 16-way dot product |

---

## BNSTORE - Weight Buffer Store

### Purpose
Loads packed ternary weights into an internal hardware buffer for subsequent BNSUM operations.

### Syntax
```assembly
BNSTORE(rs1, rs2)
```

### Encoding
```
.word (0x0B | (0x0 << 7) | (regnum_##rs1 << 15) | (regnum_##rs2 << 20) | (0x2 << 12) | (0x0) << 25 );
```

### Parameters
- **rs1**: Source register 1 (typically x0 for standard buffer operation)
- **rs2**: Source register 2 containing packed weights

### Operation
1. Takes packed 2-bit weights from rs2
2. Unpacks them into internal buffer
3. Buffer depth depends on BNRV version (4, 8, 16, 32 elements)

### Weight Encoding (1.5-bit)
- `00` → 0
- `01` → +1
- `11` → -1

### Example
```assembly
lw a2, 0(weight_ptr)    # Load 4 packed weights (8 bits)
BNSTORE(x0, a2)         # Store to buffer for next BNSUM
```

---

## BNSUM - BitNet Sum (Multiply-Accumulate)

### Purpose
Performs vectorized multiply-accumulate between int8 activations and buffered ternary weights.

### Syntax
```assembly
BNSUM(rd, rs1, rs2)
```

### Encoding
```
.word (0x0B | (regnum_##rd << 7) | (regnum_##rs1 << 15) | (regnum_##rs2 << 20) | (0x1 << 12) | (0x0) << 25 );
```

### Parameters
- **rd**: Destination register (receives 32-bit accumulated result)
- **rs1**: Source register 1 containing 4 int8 activations
- **rs2**: Source register 2 containing 4 int8 activations

### Operation
```
result = 0
for i = 0 to 7:
    weight = buffer[i]  // From previous BNSTORE
    activation = extract_int8(rs1, rs2, i)
    result += weight * activation
rd = result
```

### Multiply Operation (Ternary)
- If weight == 0: add 0
- If weight == +1: add activation
- If weight == -1: subtract activation

### Example
```assembly
BNSTORE(x0, a2)         # Load weights first
lw t0, 0(input_ptr)     # Load 4 int8 activations
lw t1, 4(input_ptr)     # Load next 4 int8 activations
BNSUM(t2, t0, t1)       # Compute 8-way dot product
add a5, a5, t2          # Accumulate into result
```

---

## DOTP8 - 8-way Dot Product

### Purpose
Direct 8-way dot product without separate buffering step.

### Syntax
```assembly
DOTP8(rd, rs1, rs2)
```

### Encoding
```
.word (0x0B | (regnum_##rd << 7) | (regnum_##rs1 << 15) | (regnum_##rs2 << 20) | (0x4 << 12) | (0x1) << 25 );
```

### Parameters
- **rd**: Destination register
- **rs1**: Activations (8 int8 values)
- **rs2**: Weights (pre-extracted/unpacked)

### Operation
Combines unpack and multiply-accumulate in single instruction.

---

## DOTP8X2 - 16-way Dot Product

### Purpose
Computes two 8-way dot products in parallel.

### Syntax
```assembly
DOTP8X2(rd, rs1, rs2)
```

### Encoding
```
.word (0x0B | (regnum_##rd << 7) | (regnum_##rs1 << 15) | (regnum_##rs2 << 20) | (0x5 << 12) | (0x1) << 25 );
```

---

## Usage Patterns

### Pattern 1: Simple 8-element SIMD

```assembly
__bitnetadd8:
    BNSTORE(x0, a2)       # Buffer 8 weights
    BNSUM(a0, a0, a1)     # Compute 8-way MAC
    ret
```

### Pattern 2: 16-element SIMD with Loop

```assembly
__bitnetadd16:
    BNSTORE(x0, a1)       # Buffer weights
    li a5, 0              # Accumulator
    li t3, 16             # Loop limit
loop:
    lw t0, 0(a0)          # Load activations
    lw t1, 4(a0)
    BNSUM(t2, t0, t1)     # 8-way MAC
    add a5, a5, t2        # Accumulate
    addi a0, a0, 8        # Advance pointer
    bne a3, t3, loop
    mv a0, a5             # Move result to return register
    ret
```

### Pattern 3: 32-element SIMD with Cascaded Buffering

```assembly
__bitnetadd32:
    BNSTORE(a2, a1)       # Buffer 16 weights (2 regs)
    li a3, 0
    li a5, 0
    li t3, 32
loop:
    lw t0, 0(a0)
    lw t1, 4(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2
    addi a3, a3, 8
    add a4, a3, a0
    bne a3, t3, loop
    mv a0, a5
    ret
```

---

## Register Conventions

### Input Registers
- **a0**: Activation pointer (for SIMD > 8) or activation data
- **a1**: Weight data or activation data
- **a2**: Weight data (for buffering)

### Output Registers
- **a0**: Return value (accumulated result)

### Scratch Registers
- **t0-t6**: Temporary values
- **a3-a5**: Loop counters and accumulators

---

## Performance Characteristics

### Latency
- **BNSTORE**: 1 cycle (typical)
- **BNSUM**: 3-5 cycles (depending on implementation)
- **DOTP8**: 4-6 cycles

### Throughput
- One BNSUM can issue every cycle (assuming no buffer conflicts)
- Achieves 8 MAC operations per BNSUM instruction

### Speedup vs Scalar
- **8-element SIMD**: ~5-6x speedup
- **16-element SIMD**: ~8-10x speedup (with loop overhead)
- **32-element SIMD**: ~10-12x speedup

---

## Compilation Flags

### Enable BNRV Instructions
```makefile
USE_SIMD=8        # Enable 8-element SIMD
BITNET_QUANT=3    # 1.5-bit quantization
```

### SIMD Width Options
- `USE_SIMD=4`: 4-element (minimal)
- `USE_SIMD=8`: 8-element (recommended)
- `USE_SIMD=16`: 16-element (higher throughput)
- `USE_SIMD=32`: 32-element (maximum performance)

### Quantization Options
- `BITNET_QUANT=2`: 1-bit quantization
- `BITNET_QUANT=3`: 1.5-bit quantization (default)
- `BITNET_QUANT=4`: 2-bit quantization

---

## C Interface

### Type Definitions
```c
typedef uint32_t int8x4_t;    // 4 int8 activations packed
typedef uint8_t int2x4_t;     // 4 ternary weights packed
typedef uint16_t int2x8_t;    // 8 ternary weights packed
```

### Function Declarations
```c
// 1.5-bit quantization variants
extern int32_t __bitnetadd4(int8x4_t a, int2x4_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int2x8_t w);
extern int32_t __bitnetadd16(int8_t *a, int2x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int2x16_t w1, int2x16_t w2);
```

### Usage Example
```c
void qmatmul(int8_t *input, int32_t *output, uint8_t *weight, int n, int d) {
    for (int i = 0; i < d; i++) {
        output[i] = 0;
        for (int j = 0; j < n; j += 8) {
            int addr = (i*n + j) >> 2;
            int8x4_t a1 = *(int8x4_t*)(input + j);
            int8x4_t a2 = *(int8x4_t*)(input + j + 4);
            uint16_t w = *(uint16_t*)(weight + addr);
            output[i] += __bitnetadd8(a1, a2, w);
        }
    }
}
```

---

## Hardware Requirements

### VexiiRiscv Configuration
```bash
# Generate hardware with BNRV8 extension
make gen-hw VEXIIARGS="--with-late-alu --regfile-async" BNRV=8
```

### Hardware Resources
- **BNRV4**: ~500 LUTs, 1 BRAM
- **BNRV8**: ~800 LUTs, 1 BRAM
- **BNRV16**: ~1200 LUTs, 2 BRAMs
- **BNRV32**: ~2000 LUTs, 4 BRAMs

### Clock Frequency
- Typically achieves 500 MHz @ 7nm (ASAP PDK)
- Critical path: accumulation tree in BNSUM

---

## Debugging Tips

### Common Issues

1. **Wrong weight encoding**: Ensure weights are packed correctly
   ```c
   // Correct: -1→0b11, 0→0b00, +1→0b01
   ```

2. **Buffer overflow**: BNSTORE has limited depth
   ```assembly
   # For BNRV8, don't store more than 8 weights before BNSUM
   ```

3. **Alignment**: Ensure activation pointers are 4-byte aligned
   ```c
   int8_t *aligned_ptr = (int8_t*)((uintptr_t)ptr & ~0x3);
   ```

### Verification

```c
// Check result against scalar implementation
int32_t expected = 0;
for (int i = 0; i < 8; i++) {
    int8_t w = extract_weight(weight, i);
    expected += w * activation[i];
}
assert(result == expected);
```

---

## Related Files

- **riscv/bnrv.h**: Instruction encoding macros
- **riscv/bnrv.S**: Assembly implementations
- **riscv/simd.h**: C function declarations
- **riscv/bitnet.h**: Matrix multiplication kernels
- **hw/VexiiRiscv-MiCo**: Hardware implementation (Scala/SpinalHDL)

---

## References

See [COMPUTATION_FLOW.md](COMPUTATION_FLOW.md) for detailed documentation on the complete computation pipeline.
