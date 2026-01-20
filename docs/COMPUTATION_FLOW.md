# BNRV Computation Flow Documentation

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [BitNet Quantization Scheme](#bitnet-quantization-scheme)
4. [Custom RISC-V Instructions](#custom-risc-v-instructions)
5. [Computation Pipeline](#computation-pipeline)
6. [Implementation Details](#implementation-details)
7. [Performance Optimization](#performance-optimization)

---

## Overview

BNRV (BitNet RISC-V) is a hardware-software co-design project that accelerates BitNet inference on RISC-V CPUs through custom SIMD instructions. The project implements efficient 1-bit and 1.5-bit quantized neural network inference by offloading the most compute-intensive operations to specialized hardware units.

### Key Features
- **Custom Instructions**: BNSTORE, BNSUM for BitNet matrix multiplication
- **Quantization Support**: 1-bit, 1.5-bit, and 2-bit weight quantization
- **SIMD Execution**: Vectorized operations processing 4, 8, 16, 32, or 64 elements
- **Hardware Acceleration**: 2.83x end-to-end speedup, 5.80x BitNet matmul speedup

---

## High-Level Architecture

The BNRV system consists of three main layers:

```
┌─────────────────────────────────────────────────────┐
│           Python Training Layer (bitmodel.py)        │
│  - BitLinear layers with quantization-aware training │
│  - Model export to binary format                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│           C Inference Engine (run.c, runbit.c)       │
│  - Transformer forward pass                          │
│  - BitNet matrix multiplication                      │
│  - Tokenization and sampling                         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│    RISC-V Hardware Layer (VexiiRiscv + BNRV)        │
│  - Custom instruction execution units                │
│  - Weight buffer management                          │
│  - SIMD vector operations                            │
└─────────────────────────────────────────────────────┘
```

---

## BitNet Quantization Scheme

### Weight Quantization

BitNet uses ternary/low-bit quantization to compress weights while maintaining accuracy:

#### 1.5-bit Quantization (Default)
- **Values**: {-1, 0, +1}
- **Encoding**: 2 bits per weight
  - `00` → 0
  - `01` → +1
  - `11` → -1
  - `10` → Reserved
- **Packing**: 4 weights per byte (2 bits × 4 = 8 bits)

#### 1-bit Quantization
- **Values**: {-1, +1}
- **Encoding**: 1 bit per weight
  - `0` → +1
  - `1` → -1
- **Packing**: 8 weights per byte

#### 2-bit Quantization
- **Values**: {-2, -1, 0, +1}
- **Encoding**: 2 bits per weight
  - `00` → 0
  - `01` → +1
  - `10` → -2
  - `11` → -1
- **Packing**: 4 weights per byte

### Activation Quantization

Activations are quantized to 8-bit signed integers:

```c
float act_scale(float *a, int n) {
    float max = -1;
    for (int i = 0; i < n; i++) {
        if (fabs(a[i]) > max) {
            max = fabs(a[i]);
        }
    }
    return max / 127.0;  // Scale to [-127, 127]
}

void act_quantize(float *a, int8_t *qa, float s, int n) {
    float scale = 1.0 / s;
    for (int i = 0; i < n; i++) {
        qa[i] = (int8_t)round(a[i] * scale);
    }
}
```

---

## Custom RISC-V Instructions

BNRV introduces three custom instructions under the CUSTOM0 opcode space (0x0B):

### 1. BNSTORE - Weight Buffer Store

**Assembly Syntax**: `BNSTORE(rs1, rs2)`

**Encoding**:
```
31    25|24  20|19  15|14  12|11   7|6     0
  func7 | rs2  | rs1  | func3|  rd  | opcode
  0x00  | rs2  | rs1  | 0x2  | 0x0  |  0x0B
```

**Operation**:
- Stores quantized weights from registers into an internal hardware buffer
- The buffer holds weights for subsequent BNSUM operations
- For 1.5-bit/2-bit: stores one 8-bit value (4 weights)
- For 1-bit: stores up to 32 bits (32 weights) across multiple calls

**Example**:
```assembly
lw a2, 0(weight_ptr)    # Load packed weights
BNSTORE(x0, a2)         # Store to hardware buffer
```

### 2. BNSUM - BitNet Sum/Multiply-Accumulate

**Assembly Syntax**: `BNSUM(rd, rs1, rs2)`

**Encoding**:
```
31    25|24  20|19  15|14  12|11   7|6     0
  func7 | rs2  | rs1  | func3|  rd  | opcode
  0x00  | rs2  | rs1  | 0x1  |  rd  |  0x0B
```

**Operation**:
- Performs vectorized multiply-accumulate using weights from buffer
- Unpacks ternary weights, multiplies with int8 activations, accumulates
- Processes 4-8 elements per instruction (depending on configuration)

**Computation** (1.5-bit example):
```
For each weight-activation pair (w, a):
  if w == 0b00: result += 0
  if w == 0b01: result += a
  if w == 0b11: result += -a
```

**Example**:
```assembly
lw t0, 0(input_ptr)     # Load 4 int8 activations
lw t1, 4(input_ptr)     # Load next 4 activations
BNSUM(t2, t0, t1)       # Compute dot product with buffered weights
add a5, a5, t2          # Accumulate result
```

### 3. DOTP8 - 8-way Dot Product (Optional)

**Assembly Syntax**: `DOTP8(rd, rs1, rs2)`

**Encoding**:
```
31    25|24  20|19  15|14  12|11   7|6     0
  func7 | rs2  | rs1  | func3|  rd  | opcode
  0x01  | rs2  | rs1  | 0x4  |  rd  |  0x0B
```

**Operation**:
- Direct 8-way dot product without buffering
- Used when weights can be extracted and passed directly

### 4. DOTP8X2 - 16-way Dot Product (Optional)

**Assembly Syntax**: `DOTP8X2(rd, rs1, rs2)`

**Operation**:
- Computes two 8-way dot products in parallel
- Further increases throughput for larger SIMD widths

---

## Computation Pipeline

### 1. Model Loading

```c
// Load model from binary file (runbit.c or embedded binary in run.c)
read_checkpoint(checkpoint_path, &config, &weights, ...);

// Weights are stored as:
// - Scale factor (float)
// - Packed quantized weights (uint8_t array)
typedef struct {
    float* s;        // Scale factor
    uint8_t* wq;     // Quantized weights (packed)
} BitNetWeight;
```

### 2. Token Embedding

```c
// Copy embedding vector for current token
float* content_row = w->token_embedding_table + token * dim;
memcpy(x, content_row, dim * sizeof(*x));
```

### 3. Transformer Layer Processing

Each transformer layer performs:

#### A. Attention Mechanism

```c
// 1. RMSNorm (simplified, no learnable weights)
bit_rmsnorm(s->xb, x, dim);

// 2. Quantize activations
float a_s = act_scale(s->xb, dim);
act_quantize(s->xb, qa, a_s, dim);

// 3. QKV projections using BitNet matmul
bit_matmul(s->q, qa, w->wq + l, dim, dim, a_s);
bit_matmul(s->k, qa, w->wk + l, dim, kv_dim, a_s);
bit_matmul(s->v, qa, w->wv + l, dim, kv_dim, a_s);

// 4. Apply RoPE positional encoding
for (int i = 0; i < dim; i += 2) {
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    // Rotate query and key vectors
    vec[i]   = v0 * fcr - v1 * fci;
    vec[i+1] = v0 * fci + v1 * fcr;
}

// 5. Multi-head attention computation
// 6. Output projection
bit_matmul(s->xb2, qa, w->wo + l, dim, dim, a_s);

// 7. Residual connection
for (int i = 0; i < dim; i++) {
    x[i] += s->xb2[i];
}
```

#### B. Feed-Forward Network

```c
// 1. RMSNorm
bit_rmsnorm(s->xb, x, dim);

// 2. Quantize
a_s = act_scale(s->xb, dim);
act_quantize(s->xb, qa, a_s, dim);

// 3. SwiGLU: w2(silu(w1(x)) * w3(x))
bit_matmul(s->hb, qa, w->w1 + l, dim, hidden_dim, a_s);
bit_matmul(s->hb2, qa, w->w3 + l, dim, hidden_dim, a_s);

// 4. Apply SiLU activation
for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb[i];
    val *= (1.0f / (1.0f + expf(-val)));  // SiLU
    val *= s->hb2[i];                      // Element-wise multiply
    s->hb[i] = val;
}

// 5. Output projection
bit_rmsnorm(s->hb, s->hb, hidden_dim);
a_s = act_scale(s->hb, dim);
act_quantize(s->hb, qa, a_s, dim);
bit_matmul(s->xb, qa, w->w2 + l, hidden_dim, dim, a_s);

// 6. Residual connection
for (int i = 0; i < dim; i++) {
    x[i] += s->xb[i];
}
```

### 4. BitNet Matrix Multiplication (Core Operation)

This is where custom instructions provide the most benefit:

#### Without Custom Instructions (Baseline)

```c
void bit_matmul(float* xout, int8_t* qa, BitNetWeight* w, int n, int d, float a_s) {
    uint8_t *weight = w->wq;
    float s = w->s[0] * a_s;
    
    for (int i = 0; i < d; i++) {
        int32_t qo = 0;
        for (int j = 0; j < n; j++) {
            // Extract 2-bit weight from packed byte
            uint8_t w_byte = weight[(i*n + j) >> 2];
            uint8_t w_shift = (w_byte >> (2 * (j & 0x03))) & 0x03;
            
            // Ternary multiply-accumulate
            qo += w_shift == 1 ? qa[j] : (w_shift == 3 ? -qa[j] : 0);
        }
        xout[i] = qo * s;  // Dequantize
    }
}
```

**Bottleneck**: Bit extraction and unpacking dominate execution time.

#### With Custom Instructions (Accelerated)

```c
void qmatmul(int8_t *input, int32_t *output, uint8_t *weight, int n, int d) {
    for (int i = 0; i < d; i++) {
        output[i] = 0;
        
        // Process 8 elements at a time with BNRV instructions
        for (int j = 0; j < n; j += 8) {
            int addr = (i*n + j) >> 2;
            
            // Load activations (2 x 4 int8 values)
            int8x4_t a1 = *(int8x4_t*)(input + j);
            int8x4_t a2 = *(int8x4_t*)(input + j + 4);
            
            // Load and buffer weights
            uint16_t w = *(uint16_t*)(weight + addr);
            
            // Custom instruction: unpack, multiply, accumulate
            output[i] += __bitnetadd8(a1, a2, w);
        }
    }
}
```

**Assembly (bnrv.S)**:
```assembly
__bitnetadd8:
    BNSTORE(x0, a2)       # Store weights to buffer
    BNSUM(a0, a0, a1)     # Compute 8-way dot product
    ret
```

**Hardware Operation**:
1. BNSTORE unpacks 8 ternary weights into internal buffer
2. BNSUM:
   - Unpacks 8 int8 activations from two registers
   - Multiplies each with corresponding buffered weight
   - Accumulates all products
   - Returns 32-bit result

---

## Implementation Details

### Python Training (bitmodel.py, bitlinear.py)

**Quantization-Aware Training**:
```python
class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator for gradients
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = x_norm + (activation_nquant(x_norm, 8) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
```

**Weight Export**:
```python
def export(self):
    u, s = weight_quant(self.weight)  # Quantize to {-1, 0, +1}
    weights = u.cpu().view(-1).to(torch.int8).numpy()
    
    # Pack 4 weights per byte (2 bits each)
    weights_per_byte = 4
    buffer = bytearray((len(weights) + 3) // 4)
    
    for i in range(0, len(weights), 4):
        byte_val = 0
        for j in range(4):
            if i + j >= len(weights):
                break
            w = weights[i + j]
            # Encode: -1→11, 0→00, 1→01
            bit_val = 0b11 if w == -1 else (0b01 if w == 1 else 0b00)
            byte_val |= (bit_val << (j * 2))
        buffer[i // 4] = byte_val
    
    return s, buffer  # Scale factor and packed weights
```

### C Inference (run.c, bitnet.h)

**Configuration Options** (compile-time):
- `USE_SIMD`: 0, 4, 8, 16, 32 (SIMD width)
- `BITNET_QUANT`: 2 (1-bit), 3 (1.5-bit), 4 (2-bit)
- `USE_DOTP`: Enable DOTP instructions instead of BNSTORE/BNSUM

**Key Functions**:

1. **bit_rmsnorm**: Simplified RMSNorm without learnable weights
2. **act_scale**: Find scaling factor for activation quantization
3. **act_quantize**: Quantize activations to int8
4. **qmatmul**: Core matrix multiplication with custom instructions
5. **bit_matmul**: Wrapper that calls qmatmul and dequantizes

### RISC-V Assembly (bnrv.S)

**SIMD Width Variants**:

```assembly
# 4-element SIMD
__bitnetadd4:
    BNSUM(a0, a0, a1)      # Both weights and activations in registers
    ret

# 8-element SIMD
__bitnetadd8:
    BNSTORE(x0, a2)        # Buffer weights
    BNSUM(a0, a0, a1)      # Process 8 elements
    ret

# 16-element SIMD
__bitnetadd16:
    BNSTORE(x0, a1)
    li a5, 0               # Result accumulator
    li t3, 16
loop:
    lw t0, 0(a0)           # Load activations
    lw t1, 4(a0)
    BNSUM(t2, t0, t1)      # 8-way operation
    add a5, a5, t2         # Accumulate
    addi a0, a0, 8
    bne a3, t3, loop
    mv a0, a5
    ret

# 32-element SIMD (uses cascaded buffering)
__bitnetadd32:
    BNSTORE(a2, a1)        # Buffer 16 weights (2 registers)
    # ... similar loop structure for 32 elements
```

### Hardware Implementation (VexiiRiscv)

The hardware extension adds:

1. **Weight Buffer**: Stores unpacked ternary weights
   - Configurable depth: 4, 8, 16, 32 elements
   - Automatically unpacks from 2-bit encoding

2. **Execution Unit**:
   - Unpacks int8 activations from registers
   - Performs ternary multiplication (0, ±1)
   - Tree-based accumulation
   - Returns 32-bit accumulated result

3. **Integration**: Custom instructions decoded in VexiiRiscv decode stage, executed in dedicated functional unit

---

## Performance Optimization

### Speedup Analysis

**Baseline (No BNRV)**: 79.4M cycles
- BitMatmul: 62.1M cycles (78.2%)
- Other ops: 17.3M cycles (21.8%)

**With BNRV8**: 28.0M cycles
- BitMatmul: 10.7M cycles (38.2%)
- Other ops: 17.3M cycles (61.8%)

**Speedup**: 5.80x for BitMatmul, 2.83x end-to-end

### Why Custom Instructions Help

1. **Eliminate Bit Extraction**: Hardware directly decodes packed weights
2. **Parallel Execution**: 8 multiply-accumulates per instruction
3. **Reduced Memory Traffic**: Weights buffered in hardware
4. **Simplified Control Flow**: No loops for bit manipulation

### Optimization Guidelines

1. **Choose SIMD Width**: Balance between:
   - Hardware area (larger buffer)
   - Loop overhead (fewer iterations)
   - Typical matrix dimensions

2. **Buffer Management**: For SIMD > 8, manage weight buffer explicitly with BNSTORE sequences

3. **Quantization Scheme**: 
   - 1.5-bit (default): Best accuracy/size tradeoff
   - 1-bit: Smallest model, highest speedup (8 weights/byte)
   - 2-bit: Best accuracy, larger models

4. **Compilation Flags**:
   ```bash
   # Maximum performance (floating-point + compressed)
   make sim-hw MARCH=rv32imfc BNRV=8
   
   # Minimal hardware
   make sim-hw MARCH=rv32i BNRV=4
   ```

---

## Example Workflow

### Complete Inference Flow

1. **Train Model** (Python):
   ```bash
   python train_bit.py --model_size 3M --quant_type 1.5b
   ```

2. **Export Model**:
   ```bash
   python export_bitmodel.py bit_model/checkpoint.pt
   ```

3. **Compile for RISC-V**:
   ```bash
   make -C riscv all USE_SIMD=8 MARCH=rv32imfc
   ```

4. **Run on Simulator**:
   ```bash
   make sim-hw MARCH=rv32imfc BNRV=8
   ```

### Dataflow Example (Single Token)

```
Input Token (int) 
    ↓
Token Embedding (float[dim])
    ↓
┌─────────────────────────────────┐
│ Layer 0                          │
│  ├─ RMSNorm + Quantize          │
│  ├─ Q/K/V Projection (BitMatmul)│  ← BNRV accelerated
│  ├─ RoPE + Attention            │
│  ├─ Output Projection (BitMatmul)│  ← BNRV accelerated
│  ├─ Residual                    │
│  ├─ FFN (2x BitMatmul + SwiGLU) │  ← BNRV accelerated
│  └─ Residual                    │
└─────────────────────────────────┘
    ↓
... (repeat for n_layers)
    ↓
Final RMSNorm
    ↓
Output Projection (float matmul)
    ↓
Logits → Sampling → Next Token
```

---

## References

- **Paper**: BNRV: A Lightweight SIMD Extension for Efficient BitNet Inference on RISC-V CPUs (ICCD 2025)
- **Base Code**: [llama2.c](https://github.com/karpathy/llama2.c)
- **BitNet**: [Microsoft BitNet](https://github.com/microsoft/BitNet)
- **VexiiRiscv**: [SpinalHDL VexiiRiscv](https://github.com/SpinalHDL/VexiiRiscv)

---

## Glossary

- **BitNet**: 1-bit or ternary quantized neural network
- **BNRV**: BitNet RISC-V custom instruction extension
- **BNSTORE**: Custom instruction to buffer weights
- **BNSUM**: Custom instruction for BitNet multiply-accumulate
- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Position Embedding
- **SwiGLU**: Swish-Gated Linear Unit activation
- **Ternary Quantization**: Quantization to {-1, 0, +1}
