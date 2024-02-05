#include <cub/block/block_reduce.cuh>

struct AttentionCausualMask {
    __forceinline__ __device__ bool
    operator()(int tokenId, int seqLen,
               int posId, int attLen) {
        // tokenId ↓ |<---attLen---->|
        //         0 | * * ... *     |
        //         1 | * * ... * *   |
        //         2 | * * ... * * * |
        // seqLen: 3 |---------------|
        return attLen + tokenId >= posId + seqLen;
    }
};

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
static __device__ void padding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const leading_dim) {
    auto batch_idx = blockIdx.x,
         token_idx = blockIdx.y,
         seq_len = gridDim.y,
         att_idx = threadIdx.x,
         att_len = blockDim.x;

    att += (batch_idx * seq_len + token_idx) * leading_dim;
    auto thread_data = mask(token_idx, seq_len, att_idx, att_len)
                           ? float(att[att_idx])
                           : -__FLT_MAX__;

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

    att[att_idx] = Tdata(thread_data * mean);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Tmask>
static __device__ void folding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const leading_dim,
    unsigned int const att_len) {
    auto batch_idx = blockIdx.x,
         token_idx = blockIdx.y,
         seq_len = gridDim.y;

    auto thread_offset = threadIdx.x * ITEMS_PER_THREAD;
    att += (batch_idx * seq_len + token_idx) * leading_dim + thread_offset;

    float thread_data[ITEMS_PER_THREAD];
#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        auto att_idx = thread_offset + i;
        thread_data[i] = att_idx < att_len && mask(token_idx, seq_len, att_idx, att_len)
                             ? float(att[i])
                             : -__FLT_MAX__;
    }

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
#pragma unroll
        for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
            thread_data[i] = expf(thread_data[i] - max);
        }
        auto acc = block_op.Sum(thread_data);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (auto att_idx = thread_offset + i; att_idx < att_len) {
            att[i] = Tdata(thread_data[i] * mean);
        }
    }
}
