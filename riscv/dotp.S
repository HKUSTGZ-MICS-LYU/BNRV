#include "bnrv.h"
// For BitNet SIMD Below 8 elements, the arguments are the data themselves
.global __dotp8
__dotp8:
    DOTP8(a0, a0, a1) // Dot Product a0 = a(a0) * w(a1)
    ret

.global __dotp8x2
__dotp8x2:
    DOTP8X2(a0, a0, a1) // Dot Product a0 = a(a0) * w(a1)
    ret
