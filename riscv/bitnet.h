#include <stdint.h>
#include <math.h>
#include "sim_stdlib.h"
#include "simd.h"

// Profiler Variables
long addsub4_time = 0;
long matmul_time = 0;
long rmsnorm_time = 0;
long quant_time = 0;


void qmatmul(int8_t *input, int32_t *output, uint8_t *weight, int n, int d){
    long start = time();

    #ifdef USE_DOTP
    int8_t w_buf[4];
    int8_t temp_w;

    for (int i=0; i<d; i++){
        output[i] = 0;
        for(int j=0; j<n; j+=4){
            uint8_t w = weight[(i*n + j) >> 2];
            for (int k=0; k<4; k++){
                temp_w = (w>>(k<<1)) & 0b11;
                w_buf[k] = temp_w == 0 ? 0 : (temp_w == 1 ? -1 : (temp_w == 2 ? -2 : -1));
            }
            output[i] += __dotp8(
                *(int32_t*) (input + j),
                *(int32_t*) w_buf);
        }
    }
    #else
    for (int i=0; i<d; i++){
        output[i] = 0;
        #if USE_SIMD == 0
        for(int j=0; j<n; j++){
            #if BITNET_QUANT == 2 // 1-bit Quantization
            uint8_t w = weight[(i*n + j) >> 3];
            uint8_t w_shift = (w >> (j&0b111)) & 0b1;
            output[i] += w_shift == 0 ? input[j] : -input[j];
            #else
            uint8_t w = weight[(i*n + j) >> 2];
            uint8_t w_shift = (w >> ((j&0b11)<<1)) & 0b11;
            #if BITNET_QUANT == 3 // 1.5-bit Quantization
            output[i] += w_shift == 1 ? input[j] : (w_shift == 3 ? -input[j] : 0); 
            #elif BITNET_QUANT == 4 // 2-bit Quantization
            output[i] += w_shift == 1 ? input[j] : \
                (w_shift == 2 ? -(input[j] << 1) : \
                (w_shift == 3 ? -input[j] : 0)); 
            #endif
            #endif
        }
        #else
        for(int j = 0; j < n; j += USE_SIMD){
            // TODO: Add Padding here when n < USE_SIMD
            #if BITNET_QUANT == 2
            int addr = (i*n + j) >> 3;
            #if USE_SIMD == 4
            int shift = j & 0b111 ? 0 : 4;
            output[i] += __bitnetadd4(*(int8x4_t*)(input + j), 
                weight[addr] >> shift);
            #elif USE_SIMD == 8
            output[i] += __bitnetadd8(*(int8x4_t*)(input + j), 
                *(int8x4_t*)(input + j + 4), weight[addr]);
            #elif USE_SIMD == 16
            output[i] += __bitnetadd16(input + j, *(int1x16_t*)(weight + addr));
            #elif USE_SIMD == 32
            output[i] += __bitnetadd32(input + j, *(int1x32_t*)(weight + addr), 0);
            #elif USE_SIMD == 64
            output[i] += __bitnetadd64(
                input + j, 
                *(int1x32_t*)(weight + addr), 
                *(int1x32_t*)(weight + addr + 4)
            );
            #endif
            #else
            int addr = (i*n + j) >> 2;
            #if USE_SIMD == 4
            output[i] += __bitnetadd4(*(int8x4_t*)(input + j), weight[addr]);
            #elif USE_SIMD == 8
            output[i] += __bitnetadd8(*(int8x4_t*)(input + j), 
                *(int8x4_t*)(input + j + 4), *(uint16_t*)(weight + addr));
            #elif USE_SIMD == 16
            output[i] += __bitnetadd16(input + j, *(int2x16_t*)(weight + addr));
            #elif USE_SIMD == 32
            output[i] += __bitnetadd32(input + j, 
                *(int1x32_t*)(weight+addr), 
                *(int1x32_t*)(weight+addr+4));
            #endif
            #endif
        }
        #endif
    }
    #endif

    matmul_time += time() - start;
}

void dequantize(int32_t *a, float *af, float s, int d){
    long start = time();

    for(int i = 0; i < d; i++){
        af[i] = a[i] * s;
    }

    quant_time += time() - start;
}


// BitNet Linear Forwarding Blocks

void bit_rmsnorm(float *a_out, float *a, int n){
    long start = time();
    float scale = 0;
    for (int i = 0; i < n; i++){
        scale += a[i]*a[i];
    }
    scale /= n;
    scale = 1.0f / sqrtf(scale);
    scale /= n;
    for (int i = 0; i < n; i++){
        a_out[i] = a[i]*scale;
    }
    rmsnorm_time += time() - start;
}

float act_scale(float *a, int n){

    long start = time();
    float max = -1;
    for (int i = 0; i < n; i++){
        if (fabs(a[i]) > max){
            max = fabs(a[i]);
        }
    }
    quant_time += time() - start;
    return max/127.0;
}

void act_quantize(float *a, int8_t *qa, float s, int n){

    long start = time();
    float scale = 1.0/s;
    for (int i = 0; i < n; i++){
        qa[i] = (int8_t)round(a[i]*scale);
    }
    quant_time += time() - start;
}