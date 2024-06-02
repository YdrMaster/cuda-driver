extern "C" __global__ void print_matrix(int *m, unsigned int d) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            printf("%6d", m[i * d + j]);
        }
        printf("\n");
    }
    printf("\n");
}

extern "C" __global__ void one_thread(int *dst, int const *src, unsigned int d) {
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            dst[i * d + j] = src[j * d + i];
}

extern "C" __global__ void one_block(int *dst, int const *src) {
    dst[threadIdx.y * blockDim.x + threadIdx.x] = src[threadIdx.x * blockDim.x + threadIdx.y];
}

// y x →
// ↓  0  1 |  2  3     0  4 |  8 12
//    4  5 |  6  7     1  5 |  9 13
//   ------+------ -> ------+------
//    8  9 | 10 11     2  6 | 10 14
//   12 13 | 14 15     3  7 | 11 15
extern "C" __global__ void multi_blocks(int *dst, int const *src) {
    extern __shared__ int shared_mem[];

    auto d_inner = blockDim.x,
         x_inner = threadIdx.x,
         y_inner = threadIdx.y,
         d_outer = gridDim.x * blockDim.x;

    shared_mem[y_inner * d_inner + x_inner] = src[(blockIdx.y * d_inner + threadIdx.y) * d_outer + blockIdx.x * d_inner + threadIdx.x];
    __syncthreads();
    dst[(blockIdx.x * d_inner + threadIdx.y) * d_outer + blockIdx.y * d_inner + threadIdx.x] = shared_mem[x_inner * d_inner + y_inner];
}
