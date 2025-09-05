#include <stdint.h>


// BitNet SIMD Types
// Activation
typedef uint32_t int8x4_t;

// Quantized Weight
typedef uint8_t int2x4_t;
typedef uint16_t int2x8_t;
typedef uint32_t int2x16_t;
typedef uint8_t int1x8_t;
typedef uint16_t int1x16_t;
typedef uint32_t int1x32_t;

// Custom Instruction Calls
#if BITNET_QUANT == 2
extern int32_t __bitnetadd4(int8x4_t a, int1x8_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int1x8_t w);
extern int32_t __bitnetadd16(int8_t *a, int1x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int1x32_t w, int1x32_t dummy);
extern int32_t __bitnetadd64(int8_t *a, int1x32_t w1, int1x32_t w2);
#else
extern int32_t __bitnetadd4(int8x4_t a, int2x4_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int2x8_t w);
extern int32_t __bitnetadd16(int8_t *a, int2x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int2x16_t w1, int2x16_t w2);
#endif