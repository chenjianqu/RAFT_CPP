//
// Created by chen on 2021/12/24.
//

#include <iostream>
#include <memory>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "src/common.h"
#include "src/Config.h"

#include <torch/torch.h>

#include "src/utils.h"


using namespace std;

struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};



int Build(const string &onnx_path,const string &tensorrt_path)
{
    cout<<"createInferBuilder"<<endl;

    ///创建builder
    auto builder=std::unique_ptr<nvinfer1::IBuilder,InferDeleter>(
            nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder)
        return -1;

    ///创建网络定义
    cout<<"createNetwork"<<endl;
    uint32_t flag=1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network=std::unique_ptr<nvinfer1::INetworkDefinition,InferDeleter>(
            builder->createNetworkV2(flag));
    if(!network)
        return -1;

    cout<<"createBuilderConfig"<<endl;
    auto config=std::unique_ptr<nvinfer1::IBuilderConfig,InferDeleter>(
            builder->createBuilderConfig());
    if(!config)
        return -1;

    ///创建parser
    cout<<"createParser"<<endl;
    auto parser=std::unique_ptr<nvonnxparser::IParser,InferDeleter>(
            nvonnxparser::createParser(*network,sample::gLogger.getTRTLogger()));
    if(!parser)
        return -1;

    ///读取模型文件

    cout<<"parseFromFile:"<<onnx_path<<endl;
    auto verbosity=sample::gLogger.getReportableSeverity();
    auto parsed=parser->parseFromFile(onnx_path.c_str(),static_cast<int>(verbosity));
    if(!parsed)
        return -1;

    //设置层工作空间大小
    config->setMaxWorkspaceSize(1_GiB);
    //使用FP16精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    cout<<"input shape:"<<network->getInput(0)->getName()<<" "<<network->getInput(0)->getDimensions()<<endl;
    cout<<"output shape:"<<network->getOutput(0)->getName()<<" "<<network->getOutput(0)->getDimensions()<<endl;

    cout<<"enableDLA"<<endl;

    ///DLA
    const int useDLACore=-1;
    samplesCommon::enableDLA(builder.get(),config.get(),useDLACore);

    ///构建engine
    cout<<"buildEngineWithConfig"<<endl;
    auto engine=std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network,*config),InferDeleter());

    if(!engine)
        return -1;

    cout<<"serializeModel"<<endl;
    auto serializeModel=engine->serialize();

    //将序列化模型拷贝到字符串
    std::string serialize_str;
    serialize_str.resize(serializeModel->size());
    memcpy((void*)serialize_str.data(),serializeModel->data(),serializeModel->size());
    //将字符串输出到文件中
    std::ofstream serialize_stream(tensorrt_path);
    serialize_stream<<serialize_str;
    serialize_stream.close();

    cout<<"done"<<endl;

    return 0;
}



void TestModel(const string& path){
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");

    auto CreateModel = [](std::unique_ptr<nvinfer1::IRuntime,InferDeleter> &runtime,
            std::shared_ptr<nvinfer1::ICudaEngine> &engine,
            std::unique_ptr<IExecutionContext, InferDeleter> &context,
            const string& path){
        std::string model_str;
        if(std::ifstream ifs(path);ifs.is_open()){
            while(ifs.peek() != EOF){
                std::stringstream ss;
                ss<<ifs.rdbuf();
                model_str.append(ss.str());
            }
            ifs.close();
        }
        else{
            auto msg=fmt::format("Can not open the DETECTOR_SERIALIZE_PATH:{}",path);
            throw std::runtime_error(msg);
        }

        info_s("createInferRuntime");

        ///创建runtime
        runtime=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
                nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

        info_s("deserializeCudaEngine");

        ///反序列化模型
        engine=std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(model_str.data(),model_str.size()) ,InferDeleter());

        info_s("createExecutionContext");

        ///创建执行上下文
        context=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
                engine->createExecutionContext());

        if(!context){
            throw std::runtime_error("can not create context");
        }
    };

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> fnet_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> fnet_engine;
    std::unique_ptr<IExecutionContext, InferDeleter> fnet_context;
    CreateModel(fnet_runtime,fnet_engine,fnet_context,path);

    for(int i=0;i<fnet_engine->getNbBindings();++i){
        cout<<fnet_engine->getBindingDimensions(i)<<endl;
    }

    cudaStream_t stream{};


    void *buffer[4]{};
    torch::Tensor tensor0 = torch::rand({1,3, 376, 1232});
    torch::Tensor tensor1 = torch::rand({1,3, 376, 1232});
    buffer[0] = tensor0.data_ptr();
    buffer[1] = tensor1.data_ptr();

    if(auto s=cudaMalloc(&buffer[0], 1*3*376*1232);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));

    if(auto s=cudaMalloc(&buffer[1], 1*3*376*1232);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));

    if(auto s=cudaMalloc(&buffer[2], 1*256*47*154);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));

    if(auto s=cudaMalloc(&buffer[3], 1*256*47*154);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));

    cudaMemcpy(buffer[0],tensor0.data_ptr(),1*3*376*1232,cudaMemcpyDeviceToDevice);
    cudaMemcpy(buffer[1],tensor1.data_ptr(),1*3*376*1232,cudaMemcpyDeviceToDevice);


    debug_s("forward_fnet set buffer");



    fnet_context->enqueue(1,buffer,stream, nullptr);


    for(int i=0;i<4;++i){
        cout<<buffer[i]<<endl;
    }

    auto opt=torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    auto dim0=fnet_engine->getBindingDimensions(2);
    debug_s("forward_fnet dim0:{}", dims2str(dim0));
    torch::Tensor fmat0 = torch::from_blob(
            buffer[2],{dim0.d[0],dim0.d[1],dim0.d[2],dim0.d[3]},opt).to(torch::kCUDA);

    auto dim1=fnet_engine->getBindingDimensions(3);
    debug_s("forward_fnet dim1:{}", dims2str(dim1));

    torch::Tensor fmat1 = torch::from_blob(
            buffer[3],{dim1.d[0],dim1.d[1],dim1.d[2],dim1.d[3]},opt).to(torch::kCUDA);

}


int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}\n",argv[1]);

    try{
        Config cfg(config_file);
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    fmt::print("start build fnet_onnx_path\n");
    if(Build(Config::fnet_onnx_path,Config::fnet_tensorrt_path)!=0){
        return -1;
    }
    fmt::print("start build cnet_onnx_path\n");
    if(Build(Config::cnet_onnx_path,Config::cnet_tensorrt_path)!=0){
        return -1;
    }
    fmt::print("start build update_onnx_path\n");
    if(Build(Config::update_onnx_path,Config::update_tensorrt_path)!=0){
        return -1;
    }

    //TestModel(Config::fnet_tensorrt_path);


    return 0;
}


