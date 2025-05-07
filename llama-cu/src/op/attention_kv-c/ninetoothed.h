#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

typedef struct {
    void *data;
    uint64_t *shape;
    int64_t *strides;
} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#endif // NINETOOTHED_H
