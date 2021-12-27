# RAFT_CPP



## Quick Start
### 0.Export RAFT onnx model
首先加载训练完成的模型权重：
```
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
这里通过对RAFT的每个子网络进行trace  
**1.1 Trace特征提取模块**  
在RAFT类里添加一个forward函数
```
def forward(self,image1, image2):
    with autocast(enabled=self.args.mixed_precision):
        fmap1, fmap2 = self.fnet([image1, image2])
    fmap1 = fmap1.float()
    fmap2 = fmap2.float()
    return fmap1,fmap2
```
trace
```
dummy_input1 = torch.randn(1, 3, 376, 1232,device='cuda')
dummy_input2 = torch.randn(1, 3, 376, 1232,device='cuda')
torch.onnx.export(model,(dummy_input1,dummy_input2),"kitti_fnet.onnx",opset_version=13)
```
**1.2 Trace Context模块**   
在RAFT类里添加forward()
```
def forward2(self,image1):
    return self.cnet(image1)
```
trace
```
dummy_input3 = torch.randn(1, 3, 376, 1232,device='cuda')
torch.onnx.export(model,dummy_input3,"kitti_cnet.onnx",opset_version=13)
```
**1.3 Trace更新模块**   
在RAFT类里添加forward()
```
def forward3(self,net, inp, corr, flow):
    return self.update_block(net, inp, corr, flow)
```
trace
```
net = torch.randn(1, 128, 47, 154,device='cuda')
inp = torch.randn(1, 128, 47, 154,device='cuda')
corr = torch.randn(1, 324, 47, 154,device='cuda')
flow = torch.randn(1, 2, 47, 154,device='cuda')
torch.onnx.export(model,(net, inp, corr, flow),"kitti_update.onnx")
```



### 1.Install
```

```


