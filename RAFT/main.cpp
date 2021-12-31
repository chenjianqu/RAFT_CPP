#include <iostream>
#include <memory>



#include "src/Config.h"
#include "src/utils.h"

#include "src/Pipeline.h"
#include "src/RAFT.h"
#include "src/Visualization.h"



cv::Mat ReadOneKitti(int index){
    char name[64];
    sprintf(name,"%06d.png",index);
    std::string img0_path=Config::DATASET_DIR+name;
    fmt::print("Read Image:{}\n",img0_path);
    return cv::imread(img0_path);
}

int main(int argc, char **argv) {
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];

    fmt::print("config_file:{}\n",argv[1]);
    RAFT::Ptr raft;
    Pipeline::Ptr pipe;
    try{
        Config cfg(config_file);
        raft = std::make_unique<RAFT>();
        pipe = std::make_shared<Pipeline>();
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    TicToc ticToc;

    cv::Mat img0,img1;
    img0 = ReadOneKitti(0);
    if(img0.empty()){
        cerr<<"无法读取图片"<<endl;
        return -1;
    }

    Tensor flow;

    for(int index=1; index <1000;++index)
    {
        img1 = ReadOneKitti(index);
        if(img1.empty()){
            cerr<<"Read image:"<<index<<" failure"<<endl;
            break;
        }
        ticToc.tic();

        Tensor tensor0 = pipe->process(img0);//(1,3,376, 1232),值大小从-1到1
        Tensor tensor1 = pipe->process(img1);//(1,3,376, 1232)

        debug_s("process:{} ms",ticToc.toc_then_tic());

        vector<Tensor> prediction = raft->forward(tensor0,tensor1);

        double forward_time = ticToc.toc_then_tic();

        debug_s("prediction:{} ms",forward_time);

        torch::Tensor output = (tensor1.squeeze()+1.)/2.;
        flow = prediction.back();//[1,2,h,w]
        flow = flow.squeeze();

        flow = pipe->unpad(flow);

        //cv::Mat flow_show = visual_flow_image(output,flow);
        cv::Mat flow_mat = visual_flow_image(flow);
        cv::Mat img_vis;
        cv::vconcat(img1,flow_mat,img_vis);
        cv::putText(img_vis, fmt::format("forward:{}ms",forward_time),{30,30},
                    cv::FONT_HERSHEY_SIMPLEX,1, {255,0,255},2);
        cv::imshow("flow",img_vis);
        if(auto order=(cv::waitKey(1) & 0xFF); order == 'q')
            break;
        else if(order==' ')
            cv::waitKey(0);

        img0 = img1;
    }


    return 0;
}
