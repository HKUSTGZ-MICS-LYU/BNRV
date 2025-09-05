// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <sim.h>

// typedef int size_t;

void setStats(int enable);

void exit(int error);

long time();

int printf(const char *format, ...);

int puts(char *s);
int putchar(int c);

// stdio.h
int sscanf(const char *str, const char *format, ...);
int sprintf(char *str, const char *format, ... );

// string.h
// int strlen(const char *s);
// void* memcpy(void *dest, const void *src, size_t count);
// void* memset( void *dest, int ch, size_t count );

// llama2 specific tokscanf
int tokscanf(const char* piece, unsigned char* byte_val);

// stdlib.h
// void* malloc( size_t size );
// void* calloc( size_t num, size_t size );
// void free( void *ptr );
// void* bsearch( const void *key, const void *ptr, size_t count, size_t size,
//                int (*comp)(const void*, const void*) );

//See https://github.com/zephyrproject-rtos/meta-zephyr-sdk/issues/110
//It does not interfere with the benchmark code.
unsigned long long __divdi3 (unsigned long long numerator,unsigned  long long divisor);
// int __fixsfsi(float a);
// int __unordsf2(float a, float b);

// Apporximate Math
float approx_sinf(float x);
float approx_cosf(float x);