#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __forceinline__ __device__ void padding(
    Tdata *__restrict__ y_,
    Tdata const *__restrict__ x_,
    Tdata const *__restrict__ w_,
    float const epsilon,
    unsigned int const leading_dim) {
    auto y = y_ + blockIdx.x * leading_dim + threadIdx.x;
    auto x = x_[blockIdx.x * leading_dim + threadIdx.x];
    auto w = w_[threadIdx.x];

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto acc = BlockOp(temp_storage).Reduce(x * x, cub::Sum());

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(blockDim.x) + epsilon));
    }
    __syncthreads();

    *y = rms * x * w;
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata>
static __forceinline__ __device__ void folding(
    Tdata *__restrict__ y_,
    Tdata const *__restrict__ x_,
    Tdata const *__restrict__ w,
    float const epsilon,
    unsigned int const leading_dim,
    unsigned int const items_size) {
    auto y = y_ + blockIdx.x * leading_dim;
    auto x = x_ + blockIdx.x * leading_dim;

    float thread_data[ITEMS_PER_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, thread_data, items_size, 0.f);
    }

    float squared[ITEMS_PER_THREAD];
#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        squared[i] = thread_data[i] * thread_data[i];
    }

    float acc;
    {
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        acc = BlockOp(temp_storage).Reduce(squared, cub::Sum());
    }

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(items_size) + epsilon));
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (auto j = i + threadIdx.x * ITEMS_PER_THREAD; j < items_size) {
            y[j] = rms * thread_data[i] * w[j];
        }
    }
}
