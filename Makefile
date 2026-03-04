# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
LD = ld

.PHONY: runbit compile-bnrv-test sim-hw-test clean
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
else ifeq ($(MARCH), rv32im_zicsr)
VEXIIARCH := --with-rvm
else ifeq ($(MARCH), rv32imc)
VEXIIARCH := --with-rvm --with-rvc
else ifeq ($(MARCH), rv32imc_zicsr)
VEXIIARCH := --with-rvm --with-rvc
else ifeq ($(MARCH), rv32imfc)
VEXIIARCH := --with-rvm --with-rvc --with-rvf
else ifeq ($(MARCH), rv32imfc_zicsr)
VEXIIARCH := --with-rvm --with-rvc --with-rvf
else ifeq ($(MARCH), rv32i)
VEXIIARCH :=
else ifeq ($(MARCH), rv32i_zicsr)
VEXIIARCH :=
endif

GENARGS := $(VEXIIARGS) $(VEXIIARCH) $(BNRVARGS)

RISCV_PATH ?= /opt/riscv
TEST_PROJ ?= test_bnrv_simple_bnrv$(BNRV)_$(MARCH)
TEST_SRCS := test_bnrv_simple.c sim_stdlib.c start.S bnrv.S
TEST_ELF := riscv/build/$(TEST_PROJ)/$(TEST_PROJ).elf

gen-hw:
	cd hw/VexiiRiscv-MiCo && sbt "runMain vexiiriscv.GenerateBnrv $(GENARGS)"

compile-bnrv: $(BNRV_ELF)

$(BNRV_ELF):
	make -C riscv all USE_SIMD=$(BNRV) MARCH=$(MARCH) RISCV_PATH=$(RISCV_PATH) 

sim-hw: $(BNRV_ELF)
	cd hw/VexiiRiscv-MiCo && sbt "runMain vexiiriscv.SimulateBnrv $(GENARGS) --load-elf $(shell realpath $(BNRV_ELF)) --print-stats"

compile-bnrv-test: $(TEST_ELF)

$(TEST_ELF): riscv/test_bnrv_simple.c riscv/sim_stdlib.c riscv/start.S riscv/bnrv.S riscv/Makefile
	make -C riscv clean
	make -C riscv compile PROJ_NAME=$(TEST_PROJ) OBJDIR=build/$(TEST_PROJ) SRCS="$(TEST_SRCS)" BIN_OBJS= USE_SIMD=$(BNRV) MARCH=$(MARCH) RISCV_PATH=$(RISCV_PATH)

sim-hw-test: $(TEST_ELF)
	cd hw/VexiiRiscv-MiCo && sbt "runMain vexiiriscv.SimulateBnrv $(GENARGS) --load-elf $(shell realpath $(TEST_ELF)) --print-stats"

.PHONY: clean
clean: 
	rm -f runbit
	make -C riscv clean
