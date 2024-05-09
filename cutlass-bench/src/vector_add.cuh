template<int kNumElemPerThread>
__device__ void vector_add(
    half *__restrict__ z,
    half const *__restrict__ x,
    half const *__restrict__ y,
    float const kx,
    float const ky,
    float const b,
    int const num) {
    if (auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * kNumElemPerThread; offset < num) {
        auto z_ptr = reinterpret_cast<half2 *>(z + offset);
        auto x_ptr = reinterpret_cast<half2 const *>(x + offset),
             y_ptr = reinterpret_cast<half2 const *>(y + offset);
        half2 kx2(kx, kx), ky2(ky, ky), b2(b, b);

#pragma unroll
        for (int i = 0; i < kNumElemPerThread / 2; i++) {
            *(z_ptr++) = kx2 * __ldg(x_ptr++) + ky2 * __ldg(y_ptr++) + b2;
        }
    }
}
