//
// Created by chen on 2021/12/24.
//

#include "RAFT.h"
#include "utils.h"
#include "Config.h"
#include "common.h"
#include "logger.h"

#include "Pipeline.h"

#include <torch/script.h>

using namespace torch::nn::functional;



RAFT::RAFT(){
    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    info_s("Read model param");

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

    CreateModel(fnet_runtime,fnet_engine,fnet_context,Config::fnet_tensorrt_path);
    CreateModel(cnet_runtime,cnet_engine,cnet_context,Config::cnet_tensorrt_path);
    CreateModel(update_runtime,update_engine,update_context,Config::update_tensorrt_path);


    fnet_jit =std::make_shared<torch::jit::Module>(torch::jit::load("/home/chen/CLionProjects/RAFT_CPP/weights/kitti_fnet.pt"));
}


tuple<Tensor, Tensor> RAFT::forward_fnet(Tensor &tensor0, Tensor &tensor1) {
    void *buffer[4]{};
    buffer[0] = tensor0.data_ptr();
    buffer[1] = tensor1.data_ptr();

    auto opt=torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    auto dim2=fnet_engine->getBindingDimensions(2);
    auto dim3=fnet_engine->getBindingDimensions(3);
    Tensor fmat0 = torch::zeros({dim2.d[0],dim2.d[1],dim2.d[2],dim2.d[3]},opt);
    Tensor fmat1 = torch::zeros({dim3.d[0],dim3.d[1],dim3.d[2],dim3.d[3]},opt);

    buffer[2] = fmat0.data_ptr();
    buffer[3] = fmat1.data_ptr();

    fnet_context->enqueue(1,buffer,stream, nullptr);

    //debug_s("forward_fnet enqueue");

    return {fmat0,fmat1};
}

tuple<Tensor, Tensor> RAFT::forward_fnet_jit(Tensor &tensor0, Tensor &tensor1) {
    std::vector<torch::jit::IValue> inputs={tensor0,tensor1};
    auto fmat = fnet_jit->forward(inputs).toTuple();
    return {fmat->elements()[0].toTensor(),fmat->elements()[1].toTensor()};
}


tuple<Tensor, Tensor> RAFT::forward_cnet(Tensor &tensor) {
    void *buffer_cnet[2];
    buffer_cnet[0] = tensor.data_ptr();

    static auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    auto dim1=cnet_engine->getBindingDimensions(1);
    Tensor fmat = torch::zeros({dim1.d[0],dim1.d[1],dim1.d[2],dim1.d[3]},opt);
    //debug_s("forward_cnet fmat.size:{}", dims2str(dim1));

    buffer_cnet[1] = fmat.data_ptr();

    cnet_context->enqueue(1,buffer_cnet,stream, nullptr);

    auto t_vector = torch::split_with_sizes(fmat,{128,128},1);

    auto net = torch::tanh(t_vector[0]);
    auto inp = torch::relu(t_vector[1]);

    return {net,inp};
}


tuple<Tensor, Tensor, Tensor> RAFT::forward_update(Tensor &net, Tensor &inp, Tensor &corr, Tensor &flow) {
    //input_names=["net_in","inp","corr","flow"],output_names=["net_out", "mask", "delta_flow"]

    void *buffer[7];
    buffer[update_engine->getBindingIndex("net_in")] = net.data_ptr();
    buffer[update_engine->getBindingIndex("inp")] = inp.data_ptr();
    buffer[update_engine->getBindingIndex("corr")] = corr.data_ptr();
    buffer[update_engine->getBindingIndex("flow")] = flow.data_ptr();

    static auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);

    int index_net_out = update_engine->getBindingIndex("net_out");
    auto net_size= update_engine->getBindingDimensions(index_net_out);
    Tensor net_output = torch::zeros({net_size.d[0],net_size.d[1],net_size.d[2],net_size.d[3]},opt);
    buffer[index_net_out]=net_output.data_ptr();
    //debug_s("forward_update net_output:{}", dims2str(net_size));

    int index_mask = update_engine->getBindingIndex("mask");
    auto mask_size= update_engine->getBindingDimensions(index_mask);
    Tensor up_mask = torch::zeros({mask_size.d[0],mask_size.d[1],mask_size.d[2],mask_size.d[3]},opt);
    buffer[index_mask] = up_mask.data_ptr();
    //debug_s("forward_update up_mask:{}", dims2str(mask_size));

    int index_delta = update_engine->getBindingIndex("delta_flow");
    auto delta_size= update_engine->getBindingDimensions(index_delta);
    Tensor delta_flow = torch::zeros({delta_size.d[0],delta_size.d[1],delta_size.d[2],delta_size.d[3]},opt);
    buffer[index_delta]=delta_flow.data_ptr();
    //debug_s("forward_update delta_flow:{}", dims2str(delta_size));

    update_context->enqueue(1,buffer,stream, nullptr);

    return {net_output,up_mask,delta_flow};
}


vector<Tensor> RAFT::compute_corr_pyramid(Tensor &tensor0, Tensor &tensor1) {
    int num_level = 4;

    ///计算corr张量
    auto size = tensor0.sizes();
    //debug_s("compute_corr_pyramid size:{}", dims2str(size));
    Tensor t0_view = tensor0.view({size[0],size[1],size[2]*size[3]});
    //debug_s("compute_corr_pyramid t0_view.shape:{}", dims2str(t0_view.sizes()));

    Tensor t1_view = tensor1.view({size[0],size[1],size[2]*size[3]});
    //debug_s("compute_corr_pyramid t1_view.shape:{}", dims2str(t1_view.sizes()));

    Tensor corr = torch::matmul(t0_view.transpose(1,2),t1_view);
    //debug_s("compute_corr_pyramid matmul:{}", dims2str(corr.sizes()));

    corr = corr.view({size[0],size[2],size[3],1,size[2],size[3]});
    //debug_s("compute_corr_pyramid view:{}", dims2str(corr.sizes()));

    corr = corr / std::sqrt(size[1]);

    corr = corr.reshape({size[0]*size[2]*size[3],1,size[2],size[3]});
    //debug_s("compute_corr_pyramid reshape :{}", dims2str(corr.sizes()));

    ///构造corr volume金字塔
    vector<Tensor> corr_pyramid;
    corr_pyramid.push_back(corr);

    for(int i=1;i<num_level;++i){
        static auto opt =AvgPool2dFuncOptions(2);//kernel size =2 ,stride =2;
        corr = avg_pool2d(corr,opt);
        //debug_s("compute_corr_pyramid i:{} avg_pool2d:{}", i,dims2str(corr.sizes()));
        corr_pyramid.push_back(corr);
    }

    return corr_pyramid;
}


tuple<Tensor, Tensor> RAFT::initialize_flow(Tensor &tensor) {
    auto size = tensor.sizes();
    debug_s("initialize_flow size:{}", dims2str(size));

    auto coords_grid = [&size](int batch,int h,int w){
        static auto opt = torch::TensorOptions(torch::kCUDA);
        auto coords_vector = torch::meshgrid({torch::arange(h,opt),torch::arange(w,opt)});//(h,w)
        //debug_s("initialize_flow coords_vector[0].shape:{}", dims2str(coords_vector[0].sizes()));
        auto coords = torch::stack({coords_vector[0],coords_vector[1]},0);//(2,h,w)
        //debug_s("initialize_flow stack.shape:{}", dims2str(coords.sizes()));
        return coords.unsqueeze(0).expand({batch,2,h,w});//(1,2,h,w)
    };

    auto coords0 = coords_grid(size[0],size[2]/8,size[3]/8);
    auto coords1 = coords_grid(size[0],size[2]/8,size[3]/8);

    return {coords0,coords1};
}


Tensor RAFT::index_corr_volume(Tensor &tensor,vector<Tensor> &pyramid){
    auto bilinear_sampler = [](Tensor &img,Tensor &coords){
        int H = img.sizes()[2];
        int W = img.sizes()[3];
        auto grids = coords.split_with_sizes({1,1},-1);
        Tensor xgrid = 2*grids[0]/(W-1) -1;
        Tensor ygrid = 2*grids[1]/(H-1) -1;
        Tensor grid = torch::cat({xgrid,ygrid},-1);

        static auto opt = GridSampleFuncOptions().align_corners(true);
        Tensor sample_img = grid_sample(img,grid,opt);
        return sample_img;
    };

    int r = 4;
    int num_level = 4;

    auto coords = tensor.permute({0,2,3,1});//(batch,h,w,2)
    auto size = coords.sizes();

    vector<Tensor> out_pyramid;
    for(int i=0;i<num_level;++i){
        debug_s("index_corr_volume iter:{}",i);

        auto corr = pyramid[i];
        static auto opt = torch::TensorOptions(torch::kCUDA);
        auto dx = torch::linspace(-r,r,2*r+1,opt);
        auto dy = torch::linspace(-r,r,2*r+1,opt);
        auto delta = torch::stack(torch::meshgrid({dy,dx}));//(2, 2*r+1, 2*r+1)

        auto centroid_lvl = coords.reshape({size[0]*size[1]*size[2],1,1,2}) / std::pow(2,i);//(batch*h*w,1,1,2)
        auto delta_lvl = delta.view({1,2*r+1,2*r+1,2});//(1, 2, 2*r+1, 2*r+1)
        auto coords_lvl = centroid_lvl + delta_lvl;

        //debug_s("index_corr_volume corr:{}", dims2str(corr.sizes()));
        //debug_s("index_corr_volume coords_lvl:{}", dims2str(coords_lvl.sizes()));

        corr = bilinear_sampler(corr,coords_lvl);
        //debug_s("index_corr_volume corr:{}", dims2str(corr.sizes()));

        corr = corr.view({size[0],size[1],size[2],-1});
        //debug_s("index_corr_volume corr.view:{}", dims2str(corr.sizes()));

        out_pyramid.push_back(corr);
    }
    Tensor out = torch::cat(out_pyramid,-1);
    return out.permute({0,3,1,2}).contiguous();
}






vector<Tensor> RAFT::forward(Tensor& tensor0, Tensor& tensor1) {
    TicToc tt;

    //debug_s("tensor0.shape:{}", dims2str(tensor0.sizes()));
    //debug_s("tensor1.shape:{}", dims2str(tensor1.sizes()));

    //fmat0:(1, 256, 47, 154), fmat1:(1, 256, 47, 154),
    auto [fmat0,fmat1] = forward_fnet(tensor0,tensor1);
    //auto [fmat0,fmat1] = forward_fnet_jit(tensor0,tensor1);
    //debug_s("fmat0.shape:{}", dims2str(fmat0.sizes()));
    //debug_s("fmat1.shape:{}", dims2str(fmat1.sizes()));

    debug_s("forward_fnet:{} ms",tt.toc_then_tic());

    /**
     * [7238, 1, 47, 154]
     * [7238, 1, 23, 77]
     * [7238, 1, 11, 38]
     * [7238, 1, 5, 19]
     */
    vector<Tensor> corr_pyramid = compute_corr_pyramid(fmat0,fmat1);
    debug_s("corr_pyramid:{} ms",tt.toc_then_tic());

    //for(auto &p : corr_pyramid) debug_s("corr_pyramid.shape:{}", dims2str(p.sizes()));

    /**
     * net:[1, 128, 47, 154]
     * inp:[1, 128, 47, 154]
     */
    auto [net,inp] = forward_cnet(tensor1);
    //debug_s("net.shape:{}", dims2str(net.sizes()));
    //debug_s("inp.shape:{}", dims2str(inp.sizes()));
    debug_s("forward_cnet:{} ms",tt.toc_then_tic());


    /**
     * coords0:[1, 2, 47, 154]
     * coords1:[1, 2, 47, 154]
     */
    auto [coords0,coords1] = initialize_flow(tensor1);
    debug_s("initialize_flow:{} ms",tt.toc_then_tic());

    //debug_s("coords0.shape:{}", dims2str(coords0.sizes()));
    //debug_s("coords1.shape:{}", dims2str(coords1.sizes()));

    vector<Tensor> flow_prediction;
    for(int i=0;i<12;++i){
        debug_s("{}",i);
        auto corr = index_corr_volume(coords1,corr_pyramid);
        //debug_s("corr_feat.shape:{}", dims2str(corr.sizes()));

        auto flow = coords1 - coords0;
        //debug_s("flow.shape:{}", dims2str(flow.sizes()));

        /**
         * net.size: [1, 128, 47, 154]
           up_mask.size: [1, 576, 47, 154]
           delta_flow.size: [1, 2, 47, 154]
         */
        auto [net1,up_mask,delta_flow] = forward_update(net,inp,corr,flow);
        net = net1;

        coords1 = coords1 + delta_flow;

        if(i==11){
        ///上采样
        static vector<int64_t> c_size = {8*coords1.sizes()[2],8*coords1.sizes()[3]};
        static auto opt = InterpolateFuncOptions().size(c_size).mode(torch::kBilinear).align_corners(true);
        auto flow_up = 8 * interpolate(coords1 - coords0,opt) ;//[1, 2, 376, 1232]

        //debug_s("flow_up shape:{}", dims2str(flow_up.sizes()));

        flow_prediction.push_back(flow_up);
        }
    }
    debug_s("iter all:{} ms",tt.toc_then_tic());


    return flow_prediction;
}

