template <class Tval, class Tidx>
static __device__ __forceinline__ void kernel(
    Tval *__restrict__ out,
    Tval const *__restrict__ table,
    Tidx const *__restrict__ index) {
    auto iy = blockIdx.x;
    auto ix = index[iy];
    out[iy * blockDim.x + threadIdx.x] = table[ix * blockDim.x + threadIdx.x];
}
