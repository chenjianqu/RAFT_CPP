import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

if __name__ == '__main__':
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

    #forward fnet
    # dummy_input1 = torch.randn(1, 3, 376, 1232,device='cuda')
    # dummy_input2 = torch.randn(1, 3, 376, 1232,device='cuda')
    # torch.onnx.export(model,(dummy_input1,dummy_input2),"kitti_fnet.onnx",input_names=["img0","img1"],output_names=["feat0","feat1"],opset_version=13)

    #forward cnet
    # dummy_input3 = torch.randn(1, 3, 376, 1232,device='cuda')
    # torch.onnx.export(model,dummy_input3,"kitti_cnet.onnx",opset_version=13,input_names=["img"],output_names=["feat"])

    #forward update
    net = torch.randn(1, 128, 47, 154,device='cuda')
    inp = torch.randn(1, 128, 47, 154,device='cuda')
    corr = torch.randn(1, 324, 47, 154,device='cuda')
    flow = torch.randn(1, 2, 47, 154,device='cuda')
    torch.onnx.export(model,(net, inp, corr, flow),"kitti_update.onnx",input_names=["net_in","inp","corr","flow"],\
                      output_names=["net_out", "mask", "delta_flow"])
