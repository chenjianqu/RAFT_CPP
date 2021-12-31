//
// Created by chen on 2021/12/24.
//

#ifndef RAFT_CPP_PIPELINE_H
#define RAFT_CPP_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class Pipeline {
public:
    using Ptr = std::shared_ptr<Pipeline>;

     torch::Tensor process(cv::Mat &img);
     torch::Tensor unpad(torch::Tensor &tensor);


private:

    int h_pad,w_pad;

};


#endif //RAFT_CPP_PIPELINE_H
