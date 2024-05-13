#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cute/tensor.hpp>

// reconstruction kernel for rotary embedding with cute
static __device__ void padding(half2 *__restrict__ x_, unsigned int const *__restrict__ pos_, float const theta,
                               unsigned int const leading_dim, int nt, int nh) {
  using namespace cute;

  // Compute dh and k
  int dh = blockDim.x;
  int k = threadIdx.x;

  // Make Tensor for x_ and pos using cutelass cute
  Tensor x = make_tensor(make_gmem_ptr(x_), make_shape(nt, nh, dh), make_stride(leading_dim, dh, 1));
  Tensor pos = make_tensor(make_gmem_ptr(pos_), make_shape(nt), make_stride(1));

  // get global memory tiles for each thread
  Tensor gX = local_tile(x, make_shape(Int<1>{}), make_coord(blockIdx.x, blockIdx.y, k));
  Tensor gPos = local_tile(pos, make_shape(Int<1>{}), make_coord(blockIdx.x));

  // get registers memory tiles for each thread
  Tensor rX = make_tensor_like(gX);
  Tensor rPos = make_tensor_like(gPos);

  // Copy global memory tiles to registers
  copy(gX, rX);
  copy(gPos, rPos);

  // Compute sin and cos
  float sin, cos;
  sincosf(float(rPos(0)) / powf(theta, float(k) / float(dh)), &sin, &cos);

  // Perform padding computation
  rX(0) = rX(0) * half2(cos, cos) + half2(-rX(0).y, rX(0).x) * half2(sin, sin);

  copy(rX, gX);
}
