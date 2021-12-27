//
// Created by chen on 2021/12/24.
//

#ifndef RAFT_CPP_UTILS_H
#define RAFT_CPP_UTILS_H

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <random>

#include <torch/torch.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <spdlog/logger.h>

#include <NvInfer.h>



class TicToc{
public:
    TicToc(){
        tic();
    }

    void tic(){
        start = std::chrono::system_clock::now();
    }

    double toc(){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

    double toc_then_tic(){
        auto t=toc();
        tic();
        return t;
    }

    void toc_print_tic(const char* str){
        std::cout<<str<<":"<<toc()<<" ms"<<std::endl;
        tic();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};


class utils {

};



template <typename T>
static std::string dims2str(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}


static std::string dims2str(nvinfer1::Dims list){
    std::string text= "[";
    for(int i=0;i<list.nbDims;++i){
        if (i > 0) text+= ", ";
        text += std::to_string(list.d[i]);
    }
    text += "]";
    return text;
}



inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp)
{
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string eigen2str(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j){
            text+=fmt::format("{:.2f} ",m(i,j));
        }
        if(m.rows()>1)
            text+="\n";
    }
    return text;
}


template<typename T>
inline std::string vec2str(const Eigen::Matrix<T,3,1> &vec){
    return eigen2str(vec.transpose());
}


inline cv::Scalar_<unsigned int> getRandomColor(){
    static std::default_random_engine rde;
    static std::uniform_int_distribution<unsigned int> color_rd(0,255);
    return {color_rd(rde),color_rd(rde),color_rd(rde)};
}



#endif //RAFT_CPP_UTILS_H
