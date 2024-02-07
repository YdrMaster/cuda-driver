// 每个 token 的数据由 `thread_per_token` 个 thread 拷贝
// n = batch * seq_len * thread_per_token 即需要的 thread 总数
// template<class Tdata, class Tindex>
// static __global__ void flatten(
//     Tdata *__restrict__ output,
//     Tdata const *__restrict__ data,
//     Tindex const *__restrict__ indices,
//     unsigned int n,
//     unsigned int thread_per_token) {
//     for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
//               step = blockDim.x * gridDim.x;
//          tid < n;
//          tid += step) {
//         auto token = __ldg(indices + tid / thread_per_token);
//         output[tid] = data[token * thread_per_token + tid % thread_per_token];
//     }
// }

// 每个 block 处理一个 token
template<class Tdata, class Tindex>
static __forceinline__ __device__ void kernel(
    Tdata *__restrict__ hidden_state,
    Tdata const *__restrict__ vocab,
    Tindex const *__restrict__ tokens) {
    {
        auto token = __ldg(tokens + blockIdx.x);
        hidden_state[blockIdx.x * blockDim.x + threadIdx.x] = vocab[token * blockDim.x + threadIdx.x];
    }
}

extern "C" __global__ void gather_float4(
    void *__restrict__ hidden_state,
    void const *__restrict__ vocab,
    void const *__restrict__ tokens) {
    {
        kernel(
            reinterpret_cast<float4 *>(hidden_state),
            reinterpret_cast<float4 const *>(vocab),
            reinterpret_cast<int const *>(tokens));
    }
}

extern "C" __global__ void gather_double4(
    void *__restrict__ hidden_state,
    void const *__restrict__ vocab,
    void const *__restrict__ tokens) {
    {
        kernel(
            reinterpret_cast<double4 *>(hidden_state),
            reinterpret_cast<double4 const *>(vocab),
            reinterpret_cast<int const *>(tokens));
    }
}
