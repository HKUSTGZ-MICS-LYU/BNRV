#include <stdint.h>
#include "sim_stdlib.h"
#include "simd.h"

static int8x4_t pack4(const int8_t *a) {
    return ((uint32_t)(uint8_t)a[0]) |
           ((uint32_t)(uint8_t)a[1] << 8) |
           ((uint32_t)(uint8_t)a[2] << 16) |
           ((uint32_t)(uint8_t)a[3] << 24);
}

static void fill_input(int8_t *a, int n) {
    for (int i = 0; i < n; i++) {
        int8_t v = (int8_t)(((i * 11) % 31) - 15);
        a[i] = (i & 1) ? (int8_t)-v : v;
    }
}

static void fill_codes(uint8_t *codes, int n) {
    for (int i = 0; i < n; i++) {
#if BITNET_QUANT == 2
        codes[i] = (uint8_t)(i & 0x1);
#else
        static const uint8_t pattern[4] = {1, 2, 3, 0};
        codes[i] = pattern[i & 0x3];
#endif
    }
}

static uint64_t pack_weights(const uint8_t *codes, int n) {
    uint64_t bits = 0;
    for (int i = 0; i < n; i++) {
#if BITNET_QUANT == 2
        bits |= (uint64_t)(codes[i] & 0x1) << i;
#else
        bits |= (uint64_t)(codes[i] & 0x3) << (i << 1);
#endif
    }
    return bits;
}

static int32_t apply_weight(int8_t a, uint8_t code) {
#if BITNET_QUANT == 2
    return code ? -a : a;
#elif BITNET_QUANT == 4
    return code == 1 ? a : (code == 2 ? -(a << 1) : (code == 3 ? -a : 0));
#else
    return code == 1 ? a : (code == 3 ? -a : 0);
#endif
}

static int32_t scalar_bitnet_sum(const int8_t *a, int n, uint64_t bits) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
#if BITNET_QUANT == 2
        uint8_t code = (uint8_t)((bits >> i) & 0x1);
#else
        uint8_t code = (uint8_t)((bits >> (i << 1)) & 0x3);
#endif
        sum += apply_weight(a[i], code);
    }
    return sum;
}

static int run_variant_test(void) {
#if USE_SIMD == 4
    int8_t a[4];
    uint8_t codes[4];
    fill_input(a, 4);
    fill_codes(codes, 4);
    uint64_t bits = pack_weights(codes, 4);
    int32_t ref = scalar_bitnet_sum(a, 4, bits);
#if BITNET_QUANT == 2
    int32_t got = __bitnetadd4(pack4(a), (int1x8_t)bits);
#else
    int32_t got = __bitnetadd4(pack4(a), (int2x4_t)bits);
#endif
    printf("BNRV4: ref=%d got=%d\n", ref, got);
    return ref != got;
#elif USE_SIMD == 8
    int8_t a[8];
    uint8_t codes[8];
    fill_input(a, 8);
    fill_codes(codes, 8);
    uint64_t bits = pack_weights(codes, 8);
    int32_t ref = scalar_bitnet_sum(a, 8, bits);
#if BITNET_QUANT == 2
    int32_t got = __bitnetadd8(pack4(a), pack4(a + 4), (int1x8_t)bits);
#else
    int32_t got = __bitnetadd8(pack4(a), pack4(a + 4), (int2x8_t)bits);
#endif
    printf("BNRV8: ref=%d got=%d\n", ref, got);
    return ref != got;
#elif USE_SIMD == 16
    int8_t a[16];
    uint8_t codes[16];
    fill_input(a, 16);
    fill_codes(codes, 16);
    uint64_t bits = pack_weights(codes, 16);
    int32_t ref = scalar_bitnet_sum(a, 16, bits);
#if BITNET_QUANT == 2
    int32_t got = __bitnetadd16(a, (int1x16_t)bits);
#else
    int32_t got = __bitnetadd16(a, (int2x16_t)bits);
#endif
    printf("BNRV16: ref=%d got=%d\n", ref, got);
    return ref != got;
#elif USE_SIMD == 32
    int8_t a[32];
    uint8_t codes[32];
    fill_input(a, 32);
    fill_codes(codes, 32);
    uint64_t bits = pack_weights(codes, 32);
    int32_t ref = scalar_bitnet_sum(a, 32, bits);
#if BITNET_QUANT == 2
    int32_t got = __bitnetadd32(a, (int1x32_t)bits, 0);
#else
    int32_t got = __bitnetadd32(a, (int2x16_t)bits, (int2x16_t)(bits >> 32));
#endif
    printf("BNRV32: ref=%d got=%d\n", ref, got);
    return ref != got;
#else
    printf("[FAIL] unsupported USE_SIMD=%d (expected 4/8/16/32)\n", USE_SIMD);
    return 1;
#endif
}

int main(void) {
    int errors = 0;
    int32_t arithmetic = (17 * 3) + (9 - 4);
    if (arithmetic != 56) {
        printf("[FAIL] arithmetic check got=%d expected=56\n", arithmetic);
        errors++;
    } else {
        printf("[PASS] arithmetic check\n");
    }

    errors += run_variant_test();

    if (errors) {
        printf("TEST FAILED, errors=%d\n", errors);
        exit(1);
    }
    printf("TEST PASSED\n");
    exit(0);
    return 0;
}
