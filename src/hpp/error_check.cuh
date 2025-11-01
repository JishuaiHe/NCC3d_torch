#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                            \
do                                                                        \
{                                                                         \
    const cudaError_t error_code = call;                                  \
    if (error_code != cudaSuccess)                                        \
    {                                                                     \
        printf("\n\033[31m%s\033[0m\n", "CUDA Error:");                   \
        printf("    File:        %s\n", __FILE__);                         \
        printf("    Line:        %d\n", __LINE__);                         \
        printf("    Error code:  %d\n", error_code);                       \
        printf("    Error text:  %s\n", cudaGetErrorString(error_code));   \
        printf("\n");                                                     \
        exit(1);                                                          \
    }                                                                     \
} while (0)

