# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
LD = ld

.PHONY: runbit
runbit: runbit.c
	$(CC) -O3 -o runbit runbit.c -lm


init-hw:
	git submodule update --init --recursive

BNRV ?= 0
VEXIIARGS ?=
BNRVARGS ?= --use-bnrv --bnrv-version $(BNRV)
MARCH ?= rv32imc
VEXIIARCH ?=

USE_PREBUILT ?= 0

BNRV_BUILD := riscv/build

ifeq ($(USE_PREBUILT), 1)
BNRV_BUILD := riscv/prebuilt/bnrv$(BNRV)_$(MARCH)
endif

BNRV_ELF := $(BNRV_BUILD)/bitllama.elf

ifeq ($(BNRV), 0)
BNRVARGS :=
endif

ifeq ($(MARCH), rv32im)
VEXIIARCH := --with-rvm
else ifeq ($(MARCH), rv32imc)
VEXIIARCH := --with-rvm --with-rvc
else ifeq ($(MARCH), rv32imfc)
VEXIIARCH := --with-rvm --with-rvc --with-rvf
else ifeq ($(MARCH), rv32i)
VEXIIARCH :=
endif

GENARGS := $(VEXIIARGS) $(VEXIIARCH) $(BNRVARGS)

gen-hw:
	cd hw/VexiiRiscv-MiCo && sbt "runMain vexiiriscv.GenerateBnrv $(GENARGS)"

compile-bnrv: $(BNRV_ELF)

$(BNRV_ELF):
	make -C riscv compile USE_SIMD=$(BNRV) MARCH=$(MARCH)

sim-hw: $(BNRV_ELF)
	cd hw/VexiiRiscv-MiCo && sbt "runMain vexiiriscv.SimulateBnrv $(GENARGS) --load-elf $(shell realpath $(BNRV_ELF)) --print-stats"

.PHONY: clean
clean: 
	rm -f runbit
	make -C riscv clean