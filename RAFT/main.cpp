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
    try{
        Config cfg(config_file);
        raft = std::make_unique<RAFT>();
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

    for(int index=1; index <1000;++index)
    {
        img1 = ReadOneKitti(index);
        if(img1.empty()){
            cerr<<"Read image:"<<index<<" failure"<<endl;
            break;
        }
        ticToc.tic();

        Tensor tensor0 = Pipeline::process(img0);//(1,3,376, 1232),值大小从-1到1
        Tensor tensor1 = Pipeline::process(img1);//(1,3,376, 1232)

        debug_s("process:{} ms",ticToc.toc_then_tic());

        vector<Tensor> prediction = raft->forward(tensor0,tensor1);

        debug_s("prediction:{} ms",ticToc.toc_then_tic());

        torch::Tensor output = (tensor1.squeeze()+1.)/2.;
        Tensor flow = prediction.back();//[1,2,h,w]
        flow = flow.squeeze();

        string msg;
        int cnt=0;
        for(int i=0;i<flow.sizes()[1];++i){
            for(int j=0;j<flow.sizes()[2];++j){
                if(i==j){
                    cnt++;
                    msg+=fmt::format("({},{}:{:.2f},{:.2f})  ",i,j,
                                     flow.index({0,i,j}).item().toFloat(),
                                     flow.index({1,i,j}).item().toFloat());
                    if(cnt%5==0)msg+="\n";
                }
            }
        }
        debug_s(msg);


        cv::Mat flow_show = visual_flow_image(output,flow);
        //cv::Mat flow_show = visual_flow_image(flow);

        cv::imshow("flow",flow_show);
        if(auto order=(cv::waitKey(0) & 0xFF); order == 'q')
            break;
        else if(order==' ')
            cv::waitKey(0);

        img0 = img1;
    }


    return 0;
}
