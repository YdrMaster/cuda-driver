#include <cuda_fp16.h>

static __device__ void rope(
    half2 *__restrict__ x_,
    unsigned int const *__restrict__ pos_,
    float const theta,
    unsigned int const leading_dim,
    int nt,
    int nh) {

    auto dh = blockDim.x;
    auto k = threadIdx.x;

    auto &x = x_[blockIdx.x * leading_dim + blockIdx.y * dh + k];
    auto pos = float(pos_[blockIdx.x]);

    float sin, cos;
    sincosf(pos / powf(theta, float(k) / float(dh)), &sin, &cos);

    x = x * half2(cos, cos) + half2(-x.y, x.x) * half2(sin, sin);
}
