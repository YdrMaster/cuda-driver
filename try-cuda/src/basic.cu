extern "C" __global__ void print(int *arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", arr[idx]);
}
