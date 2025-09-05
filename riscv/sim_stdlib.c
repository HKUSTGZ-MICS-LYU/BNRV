// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

#include "sim_stdlib.h"

void setStats(int enable)
{

}

void exit(int error){
    extern void pass();
    extern void fail();
    if(error) fail(); else pass();
    while(1);
}

long time(){
    return sim_time();
}


static void printf_c(int c)
{
    putchar(c);
}

static void printf_s(char *p)
{
    while (*p)
        putchar(*(p++));
}

static void printf_d(int val)
{
    char buffer[32];
    char *p = buffer;
    if (val < 0) {
        printf_c('-');
        val = -val;
    }
    while (val || p == buffer) {
        *(p++) = '0' + val % 10;
        val = val / 10;
    }
    while (p != buffer)
        printf_c(*(--p));
}
static printf_x(uint32_t val)
{
    char buffer[32];
    char *p = buffer;
    printf_c('0');
    printf_c('x');
    while (val || p == buffer) {
        int mod = val % 16;
        if(mod < 10){
            *(p++) = '0' + mod;
        }else{
            *(p++) = 'A' + mod - 10;
        }
        val = val / 16;
    }
    while (p != buffer)
        printf_c(*(--p));
}

int printf(const char *format, ...)
{
    int i;
    va_list ap;

    va_start(ap, format);

    for (i = 0; format[i]; i++)
        if (format[i] == '%') {
            while (format[++i]) {
                if (format[i] == 'c') {
                    printf_c(va_arg(ap,int));
                    break;
                }
                if (format[i] == 's') {
                    printf_s(va_arg(ap,char*));
                    break;
                }
                if (format[i] == 'd') {
                    printf_d(va_arg(ap,int));
                    break;
                }
                if (format[i] == 'x') {
                    printf_x(va_arg(ap,uint32_t));
                    break;
                }
            }
        } else
            printf_c(format[i]);

    va_end(ap);
}


int puts(char *s){
  while (*s) {
    putchar(*s);
    s++;
  }
  putchar('\n');
  return 0;
}

int putchar(int c){
    return sim_putchar(c);
}

int sprintf(char *str, const char *format, ... ){
    int i;
    char* str_ptr = str;
    va_list ap;

    va_start(ap, format);
    for (i = 0; format[i]; i++)
        if (format[i] == '%') {
            while (format[++i]) {
                if (format[i] == 's') {
                    char* s = va_arg(ap,char*);
                    while (*s) {
                        *str_ptr = *s;
                        str_ptr++;
                        s++;
                    }
                    break;
                }else{
                    // other formats are not supported for now
                }
            }
        } else{
            *str_ptr = format[i];
            str_ptr++;
        }
    *str_ptr = '\0';
    va_end(ap);
    return 0;
}
int sscanf(const char *str, const char *format, ...);

// llama2 tokenizer

static int hex2int(char c){
    if(c >= '0' && c <= '9'){
        return c - '0';
    }else if(c >= 'a' && c <= 'f'){
        return c - 'a' + 10;
    }else if(c >= 'A' && c <= 'F'){
        return c - 'A' + 10;
    }
    return 0;
}

// llama2 tokenizer
int tokscanf(const char* piece, unsigned char* byte_val){
    // const char* pat = "<0x00>";
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>'){
        return 0;
    }
    char h = piece[3];
    char l = piece[4];
    // convert the hex char to int
    *byte_val = (hex2int(h) << 4) | hex2int(l);
    return 1;
}
//See https://github.com/zephyrproject-rtos/meta-zephyr-sdk/issues/110
//It does not interfere with the benchmark code.
unsigned long long __divdi3 (unsigned long long numerator,unsigned  long long divisor)
{
    unsigned long long result = 0;
    unsigned long long count = 0;
    unsigned long long remainder = numerator;

    while((divisor & 0x8000000000000000ll) == 0) {
        divisor = divisor << 1;
        count++;
    }
    while(remainder != 0) {
        if(remainder >= divisor) {
            remainder = remainder - divisor;
            result = result | (1 << count);
        }
        if(count == 0) {
            break;
        }
        divisor = divisor >> 1;
        count--;
    }
    return result;
}


// code snippet from Apple Open Source 
// https://opensource.apple.com/source/gcc_os/gcc_os-1671/gcc/config/ns32k/__unordsf2.c.auto.html
# define ISNAN(x) (						\
  {								\
    union u { float f; unsigned int i; } *t = (union u *)&(x);	\
    ((t->i & 0x7f800000) == 0x7f800000) &&			\
    ((t->i & 0x7fffff) != 0);					\
  })
int __unordsf2 ( float a, float b)
{
  return ISNAN(a) || ISNAN(b);
}


int strlen(const char *s){
    int i = 0;
    while(s[i] != '\0'){
        i++;
    }
    return i;
}

// void* memcpy(void *dest, const void *src, size_t count){
//     // TODO: implement this
//     return dest;
// }
// void* memset( void *dest, int ch, size_t count ){
//     // TODO: implement this
//     return NULL;
// }

// void* malloc( size_t size ){
//     // TODO: implement this
//     return NULL;
// };
// void* calloc( size_t num, size_t size ){
//     // TODO: implement this

//     return NULL;
// };
// void free( void *ptr ){
//     // TODO: implement this
//     ptr = NULL;
// };

void *_sbrk (incr)
     int incr;
{
    extern char   end; /* Set by linker.  */
    static char * heap_end;
    char *        prev_heap_end;

    if (heap_end == 0)
      heap_end = & end;

    prev_heap_end = heap_end;
    heap_end += incr;
    printf("sbrk: %d, curr_heap_end: %x, prog_end: %x\n", incr, heap_end, &end);
    printf("Current Heap Usage: %d KB\n", (heap_end - &end) / 1024);
    return (void *) prev_heap_end;
}

void* bsearch( const void *key, const void *ptr, size_t count, size_t size,
               int (*comp)(const void*, const void*) ){
  
    const char *base = ptr;
    size_t lim;
    int cmp;
    const void *p;

    for(lim = count; lim !=0; lim >>=1){
        p = base + (lim >> 1) * size;
        cmp = comp(key, p);
        if (cmp == 0){
            return (void *)p;
        }
        if (cmp > 0){
            base = (char*)p + size;
            lim --;
        }
    }
    return (void*)NULL;
}

// Apporximate Math

float approx_sinf(float x){
    return 0;
}

float approx_cosf(float x){
    return 0;
}