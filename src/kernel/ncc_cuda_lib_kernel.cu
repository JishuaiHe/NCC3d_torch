#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <assert.h>
#include "hpp/error_check.cuh"
#include <cublas_v2.h>

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32

__host__ __device__ __forceinline__ inline auto DIVUP(int m, int n){

    return (m + n - 1) / n;
}

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])

template<typename scalar_t>
void __global__ ncc_forward_lib_kernel(
    scalar_t* z_block,
    scalar_t* x_block,
    scalar_t* xc_block
)
{



}

// torch::Tensor img2col_3d(
//     torch::Tensor z_block,
//     torch::Tensor x_block,
//     scalar_t* z_block_reshape,
//     scalar_t* x_block_reshape
// )
// /*
//     z_block shape: (N, C, w_z, h_z, d_z)
//     x_block shape: (N, C, w_x, h_x, d_x)

//     z_block_reshape shape: (N * C, w_z * h_z * d_z)
//     x_block_reshape shape: (N * C, (w_x - w_z + 1) * (w_x - w_z + 1) * (d_x - d_z + 1), w_z * h_z * d_z)

// */
// {


// }


torch::Tensor ncc_forward_lib(    
    torch::Tensor z_block,
    torch::Tensor x_block
)
    
{
    
    
} 
