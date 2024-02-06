#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

// assert BLOCK_SIZE >= blockDim.x
template<class Tcompute, unsigned int BLOCK_SIZE, class Tdata>
static __device__ void padding(
    Tdata const *__restrict__ x_,
    Tdata *__restrict__ y_,
    unsigned int const leading_dim) {
    auto x = x_ + blockIdx.x * leading_dim;
    auto y = y_ + blockIdx.x;

    using BlockOp = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto acc = BlockOp(temp_storage).Reduce(x[threadIdx.x], cub::Sum());

    if (threadIdx.x == 0) {
        *y = Tdata(acc / Tcompute(blockDim.x));
    }
}

template<class Tcompute, unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata>
static __device__ void folding(
    Tdata const *__restrict__ x_,
    Tdata *__restrict__ y_,
    unsigned int const leading_dim,
    unsigned int const item_size) {
    auto x = x_ + blockIdx.x * leading_dim;
    auto y = y_ + blockIdx.x;

    Tcompute thread_data[ITEMS_PER_THREAD];
    {
        using BlockOp = cub::BlockLoad<Tcompute, BLOCK_SIZE, ITEMS_PER_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, thread_data, item_size, 0.f);
    }
    Tcompute acc;
    {
        using BlockOp = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        acc = BlockOp(temp_storage).Reduce(thread_data, cub::Sum());
    }

    if (threadIdx.x == 0) {
        *y = Tdata(acc / Tcompute(item_size));
    }
}
