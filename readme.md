# RAFT_CPP
这是光流算法RAFT的C++实现，基于**Libtorch** + **TensorRT**。


## Quick Start
### 0.Run RAFT  
首先下载[RAFT](https://github.com:chenjianqu/RAFT)的源码，并运行成功。

### 1.Export ONNX Model
RAFT内部有3个带参数的子网络，这里分别导出。为此，这里通过对每个子网络编写forward()函数实现。
**1.1 `export_onnx.py`加载权重：**   
首先加载训练完成的模型权重：
```python
parser = argparse.ArgumentParser()
parser.add_argument('--model',default="models/raft-kitti.pth", help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))

model = model.module
model.to("cuda")
model.eval()
```

**1.1 导出特征提取模块**  
首先注释原来的forward()函数，并在RAFT类里添加一个forward函数
```python
def forward(self, image1:torch.Tensor, image2:torch.Tensor):
    fmap1, fmap2 = self.fnet([image1, image2],True)
    fmap1 = fmap1.float()
    fmap2 = fmap2.float()
    return fmap1, fmap2
```
在`export_onnx.py`中编写trace代码
```python
dummy_input1 = torch.randn(1, 3, 376, 1232,device='cuda')
dummy_input2 = torch.randn(1, 3, 376, 1232,device='cuda')
torch.onnx.export(model,(dummy_input1,dummy_input2),"kitti_fnet.onnx",opset_version=13)
```
**1.2 Trace Context模块**   
首先注释原来的forward()函数，并在RAFT类里添加一个forward函数
```python
def forward(self, image1:torch.Tensor):
    return self.cnet([image1],False)[0]
```
在`export_onnx.py`中编写trace代码
```python
dummy_input3 = torch.randn(1, 3, 376, 1232,device='cuda')
torch.onnx.export(model,dummy_input3,"kitti_cnet.onnx",opset_version=13)
```
**1.3 Trace更新模块**   
在RAFT类里添加forward()
```python
def forward(self,net, inp, corr, flow):
    return self.update_block(net, inp, corr, flow)
```
在`export_onnx.py`中编写trace代码
```python
net = torch.randn(1, 128, 47, 154,device='cuda')
inp = torch.randn(1, 128, 47, 154,device='cuda')
corr = torch.randn(1, 324, 47, 154,device='cuda')
flow = torch.randn(1, 2, 47, 154,device='cuda')
torch.onnx.export(model,(net, inp, corr, flow),"kitti_update.onnx")
```



### 2.下载并编译项目
```shell
git clone https://github.com/chenjianqu/RAFT_CPP.git

cd RAFT_CPP
mkdir build && cd build
```
修改`CMakeLists.txt`文件，以确保正确的编译。  
编译：
```shell
cmake ..
make -j10
```

### 运行
首先修改配置文件，`config.yaml`：
```yaml
%YAML:1.0

fnet_onnx_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_fnet.onnx"
fnet_tensorrt_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_fnet.bin"

cnet_onnx_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_cnet.onnx"
cnet_tensorrt_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_cnet.bin"

update_onnx_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_update.onnx"
update_tensorrt_path: "/home/chen/CLionProjects/RAFT_CPP/weights/kitti_update.bin"



segmentor_log_path: "segmentor_log.txt"
segmentor_log_level: "debug"
segmentor_log_flush: "debug"

DATASET_DIR: "/home/chen/CLionProjects/RAFT_CPP/demo/kitti07/"
WARN_UP_IMAGE_PATH: "/home/chen/CLionProjects/InstanceSegment/config/kitti.png"
```
然后编译onnx模型：
```shell
./build ../config/config.yaml
```

   
运行：
```shell
./RAFT_CPP ../config/config.yaml
```