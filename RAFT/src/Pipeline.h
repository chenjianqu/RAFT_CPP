//
// Created by chen on 2021/12/24.
//

#ifndef RAFT_CPP_PIPELINE_H
#define RAFT_CPP_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class Pipeline {
public:


    static torch::Tensor process(cv::Mat &img);


private:

};


#endif //RAFT_CPP_PIPELINE_H
