#include <cute/tensor.hpp>

template<int kNumElemPerThread>
__global__ void vector_add(
    half *__restrict__ z,
    half const *__restrict__ x,
    half const *__restrict__ y,
    float const kx,
    float const ky,
    float const b,
    int const num) {
    using namespace cute;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num / kNumElemPerThread) {
        return;
    }

    Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
    Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
    Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

    Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

    Tensor tzR = make_tensor_like(tzr);
    Tensor txR = make_tensor_like(txr);
    Tensor tyR = make_tensor_like(tyr);

    // LDG.128
    copy(txr, txR);
    copy(tyr, tyR);

    auto tzR2 = recast<half2>(tzR),
         txR2 = recast<half2>(txR),
         tyR2 = recast<half2>(tyR);
    half2 kx2(kx, kx), ky2(ky, ky), b2(b, b);

#pragma unroll
    for (int i = 0; i < size(tzR2); ++i) {
        // two hfma2 instruction
        tzR2(i) = txR2(i) * kx2 + (tyR2(i) * ky2 + b2);
    }

    auto tzRx = recast<half>(tzR2);

    // STG.128
    copy(tzRx, tzr);
}
