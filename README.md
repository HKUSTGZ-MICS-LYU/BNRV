# BNRV: Lightweight BitNet Acceleration on RISC-V via Custom Insturctions

## Notice

This project was developed when the first version of BitNet just came out, and the `runbit.c` is only a naive implementation that just works (forked from [llama2.c](https://github.com/karpathy/llama2.c)). There have been significantly better C/C++ kernels for serious [BitNet inference](https://github.com/microsoft/BitNet). This project aims at the bare-metal implementation to be run on RISC-V cores and their RTL simulators.

## Run BitNet on your CPU

```shell
make runbit
./runbit bit_model/bit_model_1M.bin # Run 1M BitNet LLaMa2 trained on tinystories
./runbit bit_model/bit_model_3M.bin # Run 3M BitNet LLaMa2 trained on tinystories
```

## Run BitNet on RISC-V CPU RTL simulator

First, you need to prepare the VexiiRiscv-based submodule:
```shell
make init-hw
```
Then, if you want to generate VexiiRiscv with BNRV extensions:
```shell
make gen-hw VEXIIARGS="<vexii args>" BNRV=< 0,4,8,16,32 >"
```
This step needs tools like JVM and SBT that VexiiRiscv requires, please refer [VexiiRiscv](https://github.com/SpinalHDL/VexiiRiscv) document for details. You can change base VexiiRiscv configs by setting `VEXIIARGS`, such as `--lsu-l1`. And you can switch BNRV implementations by setting `BRNV`, where version 4 is for buffer-free BNRV and version 8, 16, 32 are for buffered BNRV.

Finally, if you want to simulate BitNet inference on RTL simulator of Vexii and BNRV:
```shell
make sim-hw MARCH=< rv32i[m][f][c] > VEXIIARGS="<vexii args>" BNRV=< 0,4,8,16,32 >
```
This step requires: 1. RISC-V GNU toolchain (as risc-v compiler), 2. Verilator (as RTL simulator). Just install them according to the [VexiiRiscv](https://github.com/SpinalHDL/VexiiRiscv) documents and tutorials.

> [!NOTE] 
> In different RISC-V GNU version, the `MARCH` flags have different naming standard. In newer versions.

Notice that you always need to `make clean` before simulating with another `MARCH` or `BNRV`. The simulation usually takes 5~30 minutes (tested on my Intel i7-12700). It's recommended to use `MARCH=rv32imfc` for fastest simulation.

If you don't want to setup RISC-V GNU, we have prepared a few prebuilt ELFs in `riscv/prebuilt`. Set `USE_PREBUILT=1` to use them in simulation.

The default model and tokenizer is a 3M BitNet LLaMa and a 512-word tokenizer. You can replace them by replacing `riscv/bin/model.bin` and `riscv/bin/tokenizer.bin`, and run `make clean_bin && make` under `riscv`.

## Simulation Example
Baseline Result:
```shell
make clean
# Running BitNet Inference on RV32IMFC VexiiRiscv without BNRV
make sim-hw MARCH=rv32imfc VEXIIARGS="--with-late-alu --regfile-async"
```
Simulation Output:
```
[info] Total Time: 79405300
[info] Forward Time: 79077855
[info] BitMatmul Time: 62084022
[info] RMSNorm Time: 201818
[info] Quant Time: 1325638
[info] Attention Time: 106738
[info] GLU Time: 4423640
[info] RoPE Time: 9515329
[info] FMatmul Time: 1319462
```

Accelerated Result:
```shell
make clean
# Running BitNet Inference on RV32IMFC VexiiRiscv with BNRV8
make sim-hw MARCH=rv32imfc VEXIIARGS="--with-late-alu --regfile-async" BNRV=8
```
Simulation Output:
```
[info] Total Time: 28045398
[info] Forward Time: 27716955
[info] BitMatmul Time: 10704528
[info] RMSNorm Time: 202005
[info] Quant Time: 1346039
[info] Attention Time: 104220
[info] GLU Time: 4426154
[info] RoPE Time: 9515285
[info] FMatmul Time: 1319457
```

End-to-End Inference Speedup: 79405300 / 28045398 = **2.83x**

BitNet MatMul Speedup: 62084022 / 10704528 = **5.80x**

Under ASAP 7nm PDK, the VexiiRiscv (w./w.o. BNRV) can run @ 500 MHz, so the token generation speed is accelerated from 6.3 tokens/s to 17.8 tokens/s.

In general, using a better baseline setup (add caches, add branch predictors, dual-issue) will amplify the e2e speedup. The root cause of slow BitNet MatMul on baseline CPU is the heavy bit extraction/unpacking operations, while BNRV bypasses them with hardware support.

## Training BitNet on TinyStories

An example training setup is given in `train_mini.sh`. For dataset preparation, please refer to [llama2.c](https://github.com/karpathy/llama2.c).

## Acknowledgement

https://github.com/karpathy/llama2.c

https://github.com/kyegomez/BitNet

https://github.com/cpldcpu/BitNetMCU

https://github.com/microsoft/BitNet

https://github.com/pulp-platform/RVfplib

## Cite Us

[Paper Link](./BNRV_ICCD25.pdf)

```bibtex
@inproceedings{bnrv_iccd2025,
  title     = {BNRV: A Lightweight SIMD Extension for Efficient BitNet Inference on RISC-V CPUs},
  author    = {Zijun Jiang and Yangdi Lyu},
  booktitle = {The 43nd IEEE International Conference on Computer Design (ICCD)},
  year      = {2025}
}
```
