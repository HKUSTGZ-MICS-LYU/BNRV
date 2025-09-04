# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
LD = ld

.PHONY: runbit
runbit: runbit.c
	$(CC) -O3 -o runbit runbit.c -lm

clean: 
	rm -f runbit