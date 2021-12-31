//
// Created by chen on 2021/12/27.
//

#include "Visualization.h"

#include "Config.h"
#include "utils.h"


using Tensor = torch::Tensor;
using Slice = torch::indexing::Slice;

/**
 * 构造colorwheel
 * @return
 */
Tensor make_colorwheel(){
    static auto gpu = torch::TensorOptions(torch::kCUDA);

    int RY = 15,YG = 6,GC = 4,CB = 11,BM = 13,MR = 6;
    int ncols = RY + YG + GC + CB + BM + MR;

    Tensor colorwheel = torch::zeros({ncols,3},gpu);

    int col =0;
    //RY
    colorwheel.index_put_({Slice(0,RY),0},255);
    colorwheel.index_put_({Slice(0,RY),1},
                          torch::floor(255 * torch::arange(0, RY, gpu) / RY));
    col+=RY;

    //YG
    colorwheel.index_put_({Slice(col,col+YG),0},
                          255-torch::floor(255 * torch::arange(0, YG, gpu) / YG));
    colorwheel.index_put_({Slice(col,col+YG),1},255);
    col+=YG;

    //GC
    colorwheel.index_put_({Slice(col,col+GC),1},255);
    colorwheel.index_put_({Slice(col,col+GC),2},
                          torch::floor(255 * torch::arange(0, GC, gpu) / GC));
    col+=GC;

    //CB
    colorwheel.index_put_({Slice(col,col+CB),1},
                          255-torch::floor(255 * torch::arange(0, CB, gpu) / CB));
    colorwheel.index_put_({Slice(col,col+CB),2},255);
    col+=CB;

    //BM
    colorwheel.index_put_({Slice(col,col+BM),2},255);
    colorwheel.index_put_({Slice(col,col+BM),0},
                          torch::floor(255 * torch::arange(0, BM, gpu) / BM));
    col+=BM;

    //MR
    colorwheel.index_put_({Slice(col,col+MR),2},
                          255-torch::floor(255 * torch::arange(0, MR, gpu) / MR));
    colorwheel.index_put_({Slice(col,col+MR),0},255);

    return colorwheel;
}


/**
 *
 * @param u [H,W]的张量，值[0,1],type=Float32
 * @param v [H,W]的张量，值[0,1],type=Float32
 * @return [H,W,3]的张量，值[0,255],type=UInt8
 */
Tensor flow_uv_to_colors(Tensor &u,Tensor &v){
    static Tensor colorwheel = make_colorwheel();//[55,3]
    Tensor flow_img = torch::zeros({u.sizes()[0],u.sizes()[1],3},
                                   torch::TensorOptions(torch::kCUDA).dtype(torch::kUInt8));

    int ncols = colorwheel.sizes()[0];//=55

    Tensor rad = torch::sqrt(torch::square(u)+torch::square(v));//[376, 1232]

    Tensor a=torch::atan2(-v,-u)/M_PI; //[376, 1232]

    Tensor fk = (a+1)/2*(ncols-1);//[376, 1232]
    Tensor k0 = torch::floor(fk).to(torch::kLong);//[376, 1232]
    Tensor k1 = k0+1;//[376, 1232]

    Tensor book_idx = k1==ncols;//[376, 1232]
    k1.index_put_({book_idx},0);
    Tensor f =fk-k0;

    for(int i=0;i<colorwheel.sizes()[1];++i){
        Tensor tmp = colorwheel.index({"...",i});//[55]

        Tensor col0 = tmp.index({k0}) / 255.;//[376, 1232]
        Tensor col1 = tmp.index({k1}) / 255.;
        Tensor col = (1-f)*col0 + f*col1; //[376, 1232]

        Tensor idx = (rad<=1);
        col = col.masked_scatter(idx,1 - rad * (1-col));
        col = col.masked_scatter(~idx,col * 0.75);

        int ch_idx = 2-i;
        flow_img.index({"...",ch_idx}) = torch::floor(255*col);
    }

    return flow_img;
}



/**
 *
 * @param flow_uv 光流张量，shape=[2,H,W]
 * @return
 */
Tensor flow_to_image(torch::Tensor &flow_uv)
{
    Tensor u = flow_uv[0];
    Tensor v = flow_uv[1];

    //归一化uv
    Tensor rad = torch::sqrt(torch::square(u)+torch::square(v));
    float rad_max = torch::max(rad).item().toFloat();
    float epsilon = 1e-5;
    //debug_s("flow_to_image rad_max:{}",rad_max);
    u /= (rad_max+epsilon);
    v /= (rad_max+epsilon);

    return flow_uv_to_colors(u,v);
}


/**
 *
 * @param img Tensor类型的图像，shape=[3,H,W],值范围为[0,1]
 * @param flow_uv 光流张量，shape=[2,H,W]
 * @return 可视化图像
 */
cv::Mat visual_flow_image(torch::Tensor &img,torch::Tensor &flow_uv)
{
    //chw -> hwc
    Tensor image = img.permute({1,2,0});
    image = (image*255).to(torch::kUInt8);
    torch::Tensor flow = flow_to_image(flow_uv);
    torch::Tensor show_tensor = (flow*0.5 + image*0.5).to(torch::kUInt8);
    show_tensor = show_tensor.to(torch::kCPU);
    cv::Mat img_show = cv::Mat(img.sizes()[1],img.sizes()[2],CV_8UC3,show_tensor.data_ptr()).clone();
    return img_show;
}

cv::Mat visual_flow_image(torch::Tensor &flow_uv)
{
    //[H,W,3]的张量，值[0,255],type=UInt8
    torch::Tensor flow = flow_to_image(flow_uv);
    return cv::Mat(flow.sizes()[0],flow.sizes()[1],CV_8UC3,flow.to(torch::kCPU).data_ptr()).clone();
}

