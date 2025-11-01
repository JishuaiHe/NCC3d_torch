#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cmath>

// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;

#include "../hpp/error_check.cuh"

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32

__host__ __device__ __forceinline__ inline auto DIVUP(int m, int n){

    return (m + n - 1) / n;
}

__device__ __host__ __forceinline__ inline auto CONV_SIZE(int in_size, int k_size, int stride = 1){

    return (in_size - k_size) / stride + 1;
}

__device__ __host__ __forceinline__ auto compute_index(int64_t out_index, int stride, int64_t filter_size) {

    return out_index * stride;
}

template<typename scalar_t>
__global__ void ncc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> z_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> x_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> xc_block,
    int z_block_numel,
    int total_numel
){
    // auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<WARP_SIZE>(block);

    int n = block.group_index().z; // the index of batch
    int c = block.group_index().y; // the index of channel

    int idx_ele = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();

    if (idx_ele >= total_numel) return;

    int d = idx_ele % xc_block.size(4);
    int h = (idx_ele / xc_block.size(4)) % xc_block.size(3);
    int w = idx_ele / (xc_block.size(4) * xc_block.size(3));

    int w_i = compute_index(w, 1, z_block.size(2));
    int h_i = compute_index(h, 1, z_block.size(3));
    int d_i = compute_index(d, 1, z_block.size(4));

    int laneId = tile.thread_rank();

    int kIteration = DIVUP(z_block_numel, WARP_SIZE);

    // compute mean value of z_block and part of x_block

    auto mean_z = static_cast<scalar_t>(0);
    auto mean_x = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))){

            mean_z += z_block[n][c][lane_w][lane_h][lane_d];
            mean_x += x_block[n][c][w_i + lane_w][h_i + lane_h][d_i + lane_d];
        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        mean_x += tile.shfl_xor(mean_x, offset);
        mean_z += tile.shfl_xor(mean_z, offset); 

    }

    mean_x /= z_block_numel;
    mean_z /= z_block_numel; 

    // compute the denominator of NCC

    auto den_z = static_cast<scalar_t>(0);
    auto den_x = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        // if (idx_ele == 85 && n==0 && c==0) printf("%f, %f, %f, %f %d, %d, %d\n", den_z, den_x, mean_z, mean_x, z_block_numel, total_numel, col);

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))) {

            auto den_z_val = z_block[n][c][lane_w][lane_h][lane_d] - mean_z;
            auto den_x_val = x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x;

            den_z += den_z_val * den_z_val;
            den_x += den_x_val * den_x_val;

        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        den_z += tile.shfl_down(den_z, offset);
        den_x += tile.shfl_down(den_x, offset);

    }

    // if (idx_ele == 85 && n==0 && c==0) printf("%f, %f, %f, %f %d, %d\n", den_z, den_x, mean_z, mean_x, z_block_numel, total_numel);

    // compute final value

    auto num_val = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))) {

            num_val += (z_block[n][c][lane_w][lane_h][lane_d] - mean_z) * (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x);

        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        num_val += tile.shfl_down(num_val, offset);

    }

    if (laneId == 0) xc_block[n][c][w][h][d] = num_val / (sqrtf(den_z * den_x) + 1e-10);

}

torch::Tensor ncc_cuda_forward_naive(
    torch::Tensor z_block,
    torch::Tensor x_block
){

    auto batch_size = z_block.size(0);
    auto channel_size = z_block.size(1);

    int xc_block_size[3];

    int per_block_numel = 1;

    for (int i=0; i<3; i++){
        xc_block_size[i] = CONV_SIZE(x_block.size(i+2), z_block.size(i+2), 1);

        per_block_numel *= xc_block_size[i];
    }

    auto z_block_numel = z_block.size(2) * z_block.size(3) * z_block.size(4);

    const dim3 blocks(DIVUP(per_block_numel, (THREADS_PER_BLOCK / WARP_SIZE)), channel_size, batch_size);
    const dim3 threads(THREADS_PER_BLOCK);

    auto xc_block = x_block.new_zeros({batch_size, channel_size, xc_block_size[0], xc_block_size[1], xc_block_size[2]});

    AT_DISPATCH_ALL_TYPES(x_block.scalar_type(), "ncc_cuda_forwad", ([&] {
        ncc_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            z_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            x_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            xc_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            z_block_numel,
            per_block_numel
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError()); // access the last error from kernel function
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Synchronize host and device

    return xc_block;

}

template<typename scalar_t>
__global__ void ncc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> z_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> x_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> g_xc_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> g_z_block,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> g_x_block,
    int z_block_numel,
    int total_numel
){
    // auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<WARP_SIZE>(block);

    int n = block.group_index().z; // the index of batch
    int c = block.group_index().y; // the index of channel

    int idx_ele = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();

    if (idx_ele >= total_numel) return;

    int d = idx_ele % g_xc_block.size(4);
    int h = (idx_ele / g_xc_block.size(4)) % g_xc_block.size(3);
    int w = idx_ele / (g_xc_block.size(4) * g_xc_block.size(3));

    int w_i = compute_index(w, 1, z_block.size(2));
    int h_i = compute_index(h, 1, z_block.size(3));
    int d_i = compute_index(d, 1, z_block.size(4));

    int laneId = tile.thread_rank();

    int kIteration = DIVUP(z_block_numel, WARP_SIZE);

    // compute mean value of z_block and part of x_block

    auto mean_z = static_cast<scalar_t>(0);
    auto mean_x = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))){

            mean_z += z_block[n][c][lane_w][lane_h][lane_d];
            mean_x += x_block[n][c][w_i + lane_w][h_i + lane_h][d_i + lane_d];
        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        mean_x += tile.shfl_xor(mean_x, offset);
        mean_z += tile.shfl_xor(mean_z, offset); 

    }

    mean_x /= z_block_numel;
    mean_z /= z_block_numel; 

    // compute the denominator of NCC

    auto den_z = static_cast<scalar_t>(0);
    auto den_x = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        // if (idx_ele == 85 && n==0 && c==0) printf("%f, %f, %f, %f %d, %d, %d\n", den_z, den_x, mean_z, mean_x, z_block_numel, total_numel, col);

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))) {

            den_z += (z_block[n][c][lane_w][lane_h][lane_d] - mean_z) * (z_block[n][c][lane_w][lane_h][lane_d] - mean_z);
            den_x += (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x) * (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x);

        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        den_z += tile.shfl_xor(den_z, offset);
        den_x += tile.shfl_xor(den_x, offset);

    }

    // if (idx_ele == 85 && n==0 && c==0) printf("%f, %f, %f, %f %d, %d\n", den_z, den_x, mean_z, mean_x, z_block_numel, total_numel);

    // compute numerator

    auto num_val = static_cast<scalar_t>(0);

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))) {

            num_val += (z_block[n][c][lane_w][lane_h][lane_d] - mean_z) * (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x);

        }

    }

    #pragma unroll
    for (int offset=WARP_SIZE>>1; offset>0; offset>>=1){

        num_val += tile.shfl_xor(num_val, offset);

    }

    auto den_val = sqrtf(den_z * den_x) + static_cast<scalar_t>(1e-10);
    auto second_val_z = num_val / den_z;
    auto second_val_x = num_val / den_x;
    auto g_xc_val = g_xc_block[n][c][w][d][h];

    for (int i = 0; i < kIteration; i++){

        int col = laneId * kIteration + i;

        int lane_d = col % z_block.size(4);
        int lane_h = (col / z_block.size(4)) % z_block.size(3);
        int lane_w = col / (z_block.size(4) * z_block.size(3));

        if ((lane_d < z_block.size(4)) && (lane_h < z_block.size(3)) && (lane_w < z_block.size(2))) {

            auto g_z = (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x) - (z_block[n][c][lane_w][lane_h][lane_d] - mean_z) * second_val_z / den_val;
           
            auto g_x = (z_block[n][c][lane_w][lane_h][lane_d] - mean_z) - (x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i] - mean_x) * second_val_x / den_val;

            atomicAdd(&g_z_block[n][c][lane_w][lane_h][lane_d], g_z * g_xc_val);
            atomicAdd(&g_x_block[n][c][lane_w + w_i][lane_h + h_i][lane_d + d_i], g_x * g_xc_val);
            
        }   

    }


    
}


std::vector<torch::Tensor> ncc_cuda_backward_naive(
    torch::Tensor g_xc_block,
    torch::Tensor z_block,
    torch::Tensor x_block
){
    auto batch_size = z_block.size(0);
    auto channel_size = z_block.size(1);

    auto g_z_block = torch::zeros_like(z_block);
    auto g_x_block = torch::zeros_like(x_block);

    int per_block_numel = g_xc_block.size(2) * g_xc_block.size(3) * g_xc_block.size(4);
    int z_block_numel = z_block.size(2) * z_block.size(3) * z_block.size(4);

    const dim3 blocks(DIVUP(per_block_numel, (THREADS_PER_BLOCK / WARP_SIZE)), channel_size, batch_size);
    const dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(x_block.scalar_type(), "ncc_cuda_backward", ([&] {
        ncc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            z_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            x_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            g_xc_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            g_z_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            g_x_block.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            z_block_numel,
            per_block_numel
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError()); // access the last error from kernel function
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Synchronize host and device

    return {g_z_block, g_x_block};
}