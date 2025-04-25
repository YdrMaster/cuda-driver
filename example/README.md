# 推理

| 符号 | 含义         |
|:----:|:------------:|
| `+`  | 创建         |
| `o`  | 存续         |
| `m`  | 从 Host 映射 |
| `r`  | 读           |
| `w`  | 写           |
| `x`  | 销毁         |

| tensor         | graph  | tokens   | pos       | x        | x_       | attn                            | ffn        |
|:--------------:|:------:|:--------:|:---------:|:--------:|:--------:|:-------------------------------:|:----------:|
| step \ info    |        | Ttok (n) | Tpos (n)  | Ta (n,d) | Ta (n,d) | Ta (n, (nh + nkvh + nkvh) x dh) | Ta (n, di) |
| setup          |        | +m       | +m        |          |          |                                 |            |
| embedding      | kernel | rx       | o         | +w       |          |                                 |            |
| attn_norm      | kernel |          | o         | r        | +w       |                                 |            |
| attn_qkv       | kernel |          | o         | o        | rx       | +w                              |            |
| rope           | kernel |          | r         | o        |          | rw                              |            |
| attention      | graph  |          | r         | o        |          | rw                              |            |
| attn_output    | kernel |          | o         | rw       |          | rx                              |            |
| attn_allreduce | graph  |          | o         | rw       |          |                                 |            |
| ffn_norm       | kernel |          | o         | r        | +w       |                                 |            |
| ffn_up         | kernel |          | o         | o        | rx       |                                 | +w         |
| activation     | kernel |          | o         | o        |          |                                 | rw         |
| ffn_down       | kernel |          | o         | rw       |          |                                 | rx         |
| ffn_allreduce  | graph  |          | o         | rw       |          |                                 |            |
