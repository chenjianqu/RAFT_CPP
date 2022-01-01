/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_CPP. Created by chen on 2021/12/24.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

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
