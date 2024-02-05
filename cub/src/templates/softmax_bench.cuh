#include <cub/block/block_reduce.cuh>

// reference:
//     softmax0: 180.483µs
//     softmax1: 178.753µs
//     softmax2: 174.105µs
//     softmax3: 193.886µs
//     softmax4: 174.129µs
//     softmax5: 176.839µs

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

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax3(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {

    struct MaxSum {
        float max, sum;

        constexpr static __host__ __device__ __forceinline__
            MaxSum
            reduce(MaxSum a, MaxSum b) {
            if (a.max > b.max) {
                return {a.max, a.sum + b.sum * expf(b.max - a.max)};
            } else {
                return {b.max, b.sum + a.sum * expf(a.max - b.max)};
            }
        }
    };

    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;
    MaxSum thread_data{*x, 1};

    using BlockOp = cub::BlockReduce<MaxSum, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max, mean;
    {
        auto acc = block_op.Reduce(thread_data, MaxSum::reduce);
        if (threadIdx.x == 0) {
            max = acc.max;
            mean = fdividef(1, acc.sum);
        }
    }
    __syncthreads();

    *x = expf(thread_data.max - max) * mean;
}

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax4(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;
    auto mask = threadIdx.x % 8 < 4;
    auto thread_data = mask ? *x : -__FLT_MAX__;

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

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __device__ void softmax5(
    Tdata *__restrict__ x_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim + threadIdx.x;
    auto mask = threadIdx.x % 8 < 4;
    auto thread_data = *x;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(mask ? thread_data : -__FLT_MAX__, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        auto acc = block_op.Sum(thread_data = mask ? expf(thread_data - max) : 0.f);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    *x = thread_data * mean;
}
