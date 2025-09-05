# BNRV: Lightweight BitNet Acceleration on RISC-V via Custom Insturctions

## Coming Soon... :construction:

## Notice

This project is developed when the first version of BitNet just came out, and the `runbit.c` is only a naive implementation that just works (forked from [llama2.c](https://github.com/karpathy/llama2.c)). There have been significantly better C/C++ kernels for serious [BitNet inference](https://github.com/microsoft/BitNet).

## Run BitNet on your CPU

```shell
make runbit
./runbit bit_model/bit_model_1M.bin # Run 1M BitNet LLaMa2 trained on tinystories
./runbit bit_model/bit_model_3M.bin # Run 3M BitNet LLaMa2 trained on tinystories
```

## Training BitNet on TinyStories

An example training setup is given in `train_mini.sh`. For dataset preparation, please refer to [llama2.c](https://github.com/karpathy/llama2.c).

## Acknowledgement

https://github.com/karpathy/llama2.c

https://github.com/kyegomez/BitNet

https://github.com/cpldcpu/BitNetMCU

https://github.com/microsoft/BitNet

https://github.com/pulp-platform/RVfplib