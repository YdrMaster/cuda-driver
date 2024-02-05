#include <cub/block/block_reduce.cuh>

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax0(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(*x, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float sum;
    {
        auto acc = block_op.Sum(expf(*x - max));
        if (threadIdx.x == 0) { sum = acc; }
    }
    __syncthreads();

    *x = expf(*x - max) / sum;
}

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax1(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;
    auto thread_data = *x;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_data, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float sum;
    {
        auto acc = block_op.Sum(expf(thread_data - max));
        if (threadIdx.x == 0) { sum = acc; }
    }
    __syncthreads();

    *x = expf(thread_data - max) / sum;
}

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax2(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;
    auto thread_data = *x;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_data, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        auto acc = block_op.Sum(thread_data = expf(thread_data - max));
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    *x = thread_data * mean;
}
