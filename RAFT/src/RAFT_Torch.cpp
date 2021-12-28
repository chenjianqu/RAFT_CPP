//
// Created by chen on 2021/12/28.
//

#include "RAFT_Torch.h"

#include <torch/script.h>

RAFT_Torch::RAFT_Torch(){
    raft =std::make_unique<torch::jit::Module>(torch::jit::load("/home/chen/PycharmProjects/RAFT/kitti.pt"));
}


vector<Tensor> RAFT_Torch::forward(Tensor &tensor0, Tensor &tensor1) {
    auto result = raft->forward({tensor0,tensor1}).toTensorVector();
    return result;
}

