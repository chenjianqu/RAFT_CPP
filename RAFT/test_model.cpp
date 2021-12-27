#include <iostream>
#include <memory>


#include "src/Config.h"
#include "src/utils.h"

#include "src/Pipeline.h"
#include "src/RAFT.h"


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
    float time_delta=0.1;

    cv::Mat img0,img1;
    img0 = ReadOneKitti(0);
    if(img0.empty()){
        cerr<<"无法读取图片"<<endl;
        return -1;
    }

    for(int index=1;index<10;++index)
    {
        img1 = ReadOneKitti(index);
        if(img1.empty()){
            cerr<<"Read image:"<<index<<" failure"<<endl;
            break;
        }
        ticToc.tic();


        //cv::imshow("raw",img0);
        //cv::waitKey(1);

        vector<Tensor> prediction = raft->forward(img0,img1);


        img0 = img1;
    }


    return 0;
}
