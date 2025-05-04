static __forceinline__ __device__ float sigmoid(float x) {
    return fdividef(1, 1 + expf(-x));
}

template <class Tdata>
static __device__ void swiglu(
    Tdata *__restrict__ out,
    int const stride_out,
    Tdata const *__restrict__ gate_,
    int const stride_gate,
    Tdata const *__restrict__ up_,
    int const stride_up) {
    auto k = blockIdx.x * blockDim.x + threadIdx.x,
         i = blockIdx.y * stride_gate + k,
         j = blockIdx.y * stride_up + k;
    auto x = float(gate_[i]),
         y = float(up_[j]);
    out[i] = Tdata(x * sigmoid(x) * y);
}
