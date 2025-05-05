#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

// assert BLOCK_SIZE >= blockDim.x
template <unsigned int BLOCK_SIZE, class Tw, class Ta>
static __device__ void padding(
    Ta *__restrict__ y_,
    int const stride_y,
    Ta const *__restrict__ x_,
    int const stride_x,
    Tw const *__restrict__ w_,
    float const epsilon) {

    auto y = y_ + blockIdx.x * stride_y + threadIdx.x;
    float const x = x_[blockIdx.x * stride_x + threadIdx.x];
    float const w = w_[threadIdx.x];

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto acc = BlockOp(temp_storage).Reduce(x * x, cub::Sum());

    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(acc / float(blockDim.x) + epsilon);
    }
    __syncthreads();

    *y = Ta(rms * x * w);
}

template <unsigned int BLOCK_SIZE, unsigned int NUM_ITEMS_THREAD, class Tw, class Ta>
static __device__ void folding(
    Ta *__restrict__ y,
    int const stride_y,
    Ta const *__restrict__ x,
    int const stride_x,
    Tw const *__restrict__ w,
    float const epsilon,
    unsigned int const items_size) {
    y += blockIdx.x * stride_y;
    x += blockIdx.x * stride_x;

    float data[NUM_ITEMS_THREAD], weight[NUM_ITEMS_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, data, items_size, 0.f);
        BlockOp(temp_storage).Load(w, weight, items_size, 0.f);
    }

    float squared = 0;
#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i) {
        squared += data[i] * data[i];
    }

    float acc;
    {
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        acc = BlockOp(temp_storage).Reduce(squared, cub::Sum());
    }

    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(acc / float(items_size) + epsilon);
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i) {
        data[i] = rms * data[i] * weight[i];
    }

    {
        using BlockOp = cub::BlockStore<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Store(y, data, items_size);
    }
}
