#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x) 


torch::Tensor ncc_cuda_forward_lib(
    torch::Tensor z_block,
    torch::Tensor x_block
);


torch::Tensor ncc_forward_lib(    
    torch::Tensor z_block,
    torch::Tensor x_block
){  
    CHECK_INPUT(z_block);
    CHECK_INPUT(x_block);
    return ncc_cuda_forward_lib(z_block, x_block);
}

std::vector<torch::Tensor> ncc_cuda_backward_lib(
    torch::Tensor z_block,
    torch::Tensor x_block,
    torch::Tensor g_xc_block
);


std::vector<torch::Tensor> ncc_backward_lib(    
    torch::Tensor z_block,
    torch::Tensor x_block,
    torch::Tensor g_xc_block
){  
    CHECK_INPUT(z_block);
    CHECK_INPUT(x_block);
    CHECK_INPUT(g_xc_block);
    return ncc_cuda_backward_lib(z_block, x_block, g_xc_block);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

    m.def("forward_naive", &ncc_forward_lib, "normalization cross correlation foward (CUDA)");
    m.def("backward_naive", &ncc_backward_lib, "normalization cross correlation backward (CUDA)");

}