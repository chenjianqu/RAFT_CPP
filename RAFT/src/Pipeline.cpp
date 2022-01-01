/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_CPP. Created by chen on 2021/12/24.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "Pipeline.h"
#include "utils.h"
#include "Config.h"


namespace F = torch::nn::functional;


torch::Tensor Pipeline::process(cv::Mat &img)
{
    TicToc tt;

    int h = img.rows;
    int w = img.cols;

    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    auto input_tensor = torch::from_blob(img_float.data, {h,w ,3 }, torch::kFloat32).to(torch::kCUDA);

    ///预处理
    input_tensor = 2*(input_tensor/255.0f) - 1.0f;

    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);
    debug_s("setInputTensorCuda bgr->rgb:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    debug_s("setInputTensorCuda hwc->chw:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///pad
     h_pad = ((int(h/8)+1)*8 - h)%8;
     w_pad = ((int(w/8)+1)*8 - w)%8;

    //前两个数pad是2维度，中间两个数pad第1维度，后两个数pad 第0维度
    input_tensor = F::pad(input_tensor, F::PadFuncOptions({w_pad, 0, h_pad, 0, 0, 0}).mode(torch::kConstant));


    return input_tensor.unsqueeze(0).contiguous();
}

/**
 *
 * @param tensor shape:[c,h,w]
 * @return
 */
torch::Tensor Pipeline::unpad(torch::Tensor &tensor)
{
    return tensor.index({"...",
                         at::indexing::Slice(h_pad,at::indexing::None),
                         at::indexing::Slice(w_pad,at::indexing::None)});
}







