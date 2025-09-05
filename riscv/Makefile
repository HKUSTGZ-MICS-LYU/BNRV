PROJ_NAME=bitllama

DEBUG?=no
BENCH=yes

USE_SIMD?=0
BITNET_QUANT?=3

CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -DTIME -DCORE_HZ=12000000ll
CFLAGS += -DUSE_SIMD=$(USE_SIMD) -DBITNET_QUANT=$(BITNET_QUANT)
CFLAGS += -I./

SRCS = 	$(wildcard *.c) \
		$(wildcard *.S) 

BIN_OBJS = bin/model.o bin/tokenizer.o

all: bin/model.o bin/tokenizer.o compile 

bin/model.o: bin/model.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/model.o bin/model.bin

bin/tokenizer.o: bin/tokenizer.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/tokenizer.o bin/tokenizer.bin

clean_bin:
	rm bin/*.o

include common/app.mk