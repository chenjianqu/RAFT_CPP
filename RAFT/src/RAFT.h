//
// Created by chen on 2021/12/24.
//

#ifndef RAFT_CPP_RAFT_H
#define RAFT_CPP_RAFT_H

#include <memory>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <NvInfer.h>

#include <cuda_runtime_api.h>


#include "common.h"
#include "Config.h"

using torch::Tensor;


struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};


class RAFT {
public:
    using Ptr = std::unique_ptr<RAFT>;

    RAFT();

    vector<Tensor> forward(Tensor& tensor0, Tensor& tensor1);
    vector<Tensor> forward_test();

    tuple<Tensor,Tensor> forward_fnet(Tensor &tensor0,Tensor &tensor1);
    tuple<Tensor,Tensor> forward_fnet_jit(Tensor &tensor0,Tensor &tensor1);
    tuple<Tensor,Tensor> forward_cnet(Tensor &tensor1);
    tuple<Tensor,Tensor,Tensor> forward_update(Tensor &net,Tensor &inp,Tensor &corr,Tensor &flow);
    static tuple<Tensor,Tensor> initialize_flow(Tensor &tensor1);
    void compute_corr_pyramid(Tensor &tensor0, Tensor &tensor1);
    Tensor index_corr_volume(Tensor &tensor);

private:

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> fnet_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> fnet_engine;
    std::unique_ptr<IExecutionContext, InferDeleter> fnet_context;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> cnet_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> cnet_engine;
    std::unique_ptr<IExecutionContext, InferDeleter> cnet_context;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> update_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> update_engine;
    std::unique_ptr<IExecutionContext, InferDeleter> update_context;

    std::shared_ptr<torch::jit::Module> fnet_jit;
    std::shared_ptr<torch::jit::Module> cnet_jit;
    std::shared_ptr<torch::jit::Module> update_jit;

    cudaStream_t stream{};

    Tensor last_flow;
    vector<Tensor> corr_pyramid; //相关性金字塔
};


#endif //RAFT_CPP_RAFT_H
