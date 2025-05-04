#include <cuda_fp16.h>

// template <class Tp, class Ta>
// static __device__ void padding(
//     Ta *__restrict__ t,
//     int const stride_token,
//     int const stride_head,
//     Tp const *__restrict__ pos,
//     float const *__restrict__ sin_table,
//     float const *__restrict__ cos_table) {

//     auto const
//         // nt = gridDim.y,
//         // nh_h = gridDim.x,
//         nh_l
//         = blockDim.y,
//         dh_div_2 = blockDim.x,

//         it = blockIdx.y,         // token index
//         ih_h = blockIdx.x,       // head index (high)
//         ih_l = threadIdx.y,      // head index (low)
//         ih = ih_h * nh_l + ih_l, // head index
//         i = threadIdx.x;         // element index
//     // i * 2 是因为每两个一组
//     t += it * stride_token + ih * stride_head + i * 2;

//     // 获取位置索引并确保在范围内
//     Tp pos_idx = pos[it];

//     // 从表中获取sin和cos值
//     float sin_val = sin_table[pos_idx * dh_div_2 + i];
//     float cos_val = cos_table[pos_idx * dh_div_2 + i];

//     // 读取复数值并应用旋转
//     float a = (float)t[0];
//     float b = (float)t[1];
//     *t = Ta(a * cos_val - b * sin_val, a * sin_val + b * cos_val);
// }

template <class Tp, class Ta>
static __device__ void padding(
    Ta *__restrict__ y,
    int const stride_token_y,
    int const stride_head_y,
    Ta const *__restrict__ x,
    int const stride_token_x,
    int const stride_head_x,
    Tp const *__restrict__ pos,
    float const *__restrict__ sin_table,
    float const *__restrict__ cos_table) {

    auto const
        // nt = gridDim.y,
        // nh_h = gridDim.x,
        nh_l
        = blockDim.y,
        dh_div_2 = blockDim.x,

        it = blockIdx.y,         // token index
        ih_h = blockIdx.x,       // head index (high)
        ih_l = threadIdx.y,      // head index (low)
        ih = ih_h * nh_l + ih_l, // head index
        i = threadIdx.x;         // element index

    // 计算x和y的偏移量，i * 2 是因为每两个为一组
    const Ta *x_offset = x + it * stride_token_x + ih * stride_head_x + i * 2;
    Ta *y_offset = y + it * stride_token_y + ih * stride_head_y + i * 2;

    // 获取位置索引
    Tp pos_idx = pos[it];

    // 从表中获取sin和cos值
    float sin_val = sin_table[pos_idx * dh_div_2 + i];
    float cos_val = cos_table[pos_idx * dh_div_2 + i];

    // 从x读取复数值
    float a = (float)x_offset[0];
    float b = (float)x_offset[1];

    // 应用旋转并写入y
    y_offset[0] = Ta(a * cos_val - b * sin_val);
    y_offset[1] = Ta(a * sin_val + b * cos_val);
}
