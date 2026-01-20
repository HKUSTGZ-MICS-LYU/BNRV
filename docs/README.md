# BNRV Documentation

This directory contains comprehensive documentation for the BNRV (BitNet RISC-V) project.

## Available Documentation

### 📖 [Computation Flow Documentation](COMPUTATION_FLOW.md)
**For**: Developers, researchers, and anyone wanting to understand the complete system

**Contents**:
- High-level architecture overview
- BitNet quantization schemes (1-bit, 1.5-bit, 2-bit)
- Complete computation pipeline
- Custom RISC-V instruction details
- Implementation details across Python, C, and Assembly
- Performance optimization guidelines
- Complete examples and workflows

**When to use**: When you need a comprehensive understanding of how BNRV works from training to hardware execution.

---

### 📋 [Custom Instructions Quick Reference](CUSTOM_INSTRUCTIONS.md)
**For**: Assembly programmers, compiler developers, and hardware designers

**Contents**:
- Concise instruction specifications
- Encoding formats and bit layouts
- Usage patterns and examples
- Performance characteristics
- C interface definitions
- Debugging tips

**When to use**: As a quick lookup reference when writing assembly code or implementing custom instruction support.

---

## Quick Start Guide

### Understanding the System
1. Start with the **Overview** section in [COMPUTATION_FLOW.md](COMPUTATION_FLOW.md)
2. Review the **High-Level Architecture** to understand the three layers
3. Read about **BitNet Quantization** to understand how weights are compressed

### Working with Custom Instructions
1. Read the **Custom RISC-V Instructions** section in [COMPUTATION_FLOW.md](COMPUTATION_FLOW.md) for detailed explanations
2. Use [CUSTOM_INSTRUCTIONS.md](CUSTOM_INSTRUCTIONS.md) as a quick reference while coding
3. Study the usage patterns and examples

### Implementation
1. For Python training: See **Python Training** section
2. For C inference: See **C Inference** and **Computation Pipeline** sections
3. For Assembly optimization: See **RISC-V Assembly** section and the quick reference

---

## Key Concepts

### Custom Instructions
- **BNSTORE**: Buffers packed ternary weights into hardware
- **BNSUM**: Performs 8-way multiply-accumulate with buffered weights
- **DOTP8/DOTP8X2**: Direct dot product operations

### Quantization
- **Weights**: Ternary {-1, 0, +1} packed in 2 bits each
- **Activations**: 8-bit integers with dynamic scaling
- **Scale factors**: Stored per layer for dequantization

### SIMD Widths
- **BNRV4**: 4 elements per instruction
- **BNRV8**: 8 elements (recommended)
- **BNRV16**: 16 elements
- **BNRV32**: 32 elements (maximum)

---

## Code Organization

```
BNRV/
├── bitmodel.py          # PyTorch model with BitLinear layers
├── bitlinear.py         # BitLinear quantization-aware training
├── export_bitmodel.py   # Export trained model to binary
├── runbit.c             # Standalone C inference (x86)
├── riscv/
│   ├── run.c            # RISC-V inference (embedded)
│   ├── bitnet.h         # BitNet kernel implementations
│   ├── bnrv.S           # Custom instruction assembly
│   ├── bnrv.h           # Instruction encoding macros
│   └── simd.h           # C interface to custom instructions
└── hw/
    └── VexiiRiscv-MiCo  # Hardware implementation (Scala)
```

---

## Performance

### Speedup Results (3M BitNet LLaMa2 on TinyStories)

| Metric | Baseline | BNRV8 | Speedup |
|--------|----------|-------|---------|
| Total Cycles | 79.4M | 28.0M | **2.83x** |
| BitMatmul | 62.1M | 10.7M | **5.80x** |
| Token/s @ 500MHz | 6.3 | 17.8 | **2.83x** |

### Why Custom Instructions Help
1. Eliminate bit extraction overhead
2. Parallel 8-way MAC operations
3. Reduced memory bandwidth (weights buffered)
4. Simplified control flow

---

## Common Use Cases

### Use Case 1: Training a BitNet Model
```bash
# Train
python train_bit.py

# Export
python export_bitmodel.py bit_model/checkpoint.pt
```
See: [Python Training section](COMPUTATION_FLOW.md#python-training-bitmodelpy-bitlinearpy)

### Use Case 2: Running on x86 CPU
```bash
make runbit
./runbit bit_model/bit_model_3M.bin
```
See: Main README.md

### Use Case 3: RISC-V Simulation with Custom Instructions
```bash
make sim-hw MARCH=rv32imfc BNRV=8
```
See: [Example Workflow section](COMPUTATION_FLOW.md#example-workflow)

### Use Case 4: Implementing Custom Instructions in Hardware
See: [Hardware Implementation section](COMPUTATION_FLOW.md#hardware-implementation-vexiiriscv)

---

## Related Resources

### External Links
- [BNRV Paper (ICCD 2025)](../BNRV_ICCD25.pdf)
- [VexiiRiscv Documentation](https://github.com/SpinalHDL/VexiiRiscv)
- [Microsoft BitNet](https://github.com/microsoft/BitNet)
- [llama2.c](https://github.com/karpathy/llama2.c)

### Internal Files
- Main [README.md](../README.md)
- [Makefile](../Makefile) - Build and simulation targets

---

## Contributing to Documentation

When updating documentation:
1. Keep the computation flow document comprehensive but readable
2. Keep the quick reference concise and scannable
3. Add examples for complex concepts
4. Update this README when adding new documents
5. Maintain consistent formatting and structure

---

## Questions?

For questions about:
- **Theory**: See the [paper](../BNRV_ICCD25.pdf)
- **Usage**: See [COMPUTATION_FLOW.md](COMPUTATION_FLOW.md)
- **Instructions**: See [CUSTOM_INSTRUCTIONS.md](CUSTOM_INSTRUCTIONS.md)
- **Implementation**: Check the relevant source files
- **Issues**: Open an issue on GitHub
