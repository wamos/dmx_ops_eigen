import sys, os
import torch
from torch import nn

from dmx_models import image_resize, reshape_casting, mel_scale

def run_dmx_ops(input_shape, name):    
    input_var = torch.randn(*input_shape)
    if name == "mel_scale":
        model = mel_scale()
    elif name  == "reshape_casting":
        model = reshape_casting()
    elif name == "image_resize":
        model = image_resize()
    model.eval()
    for x in range(5):
        output = model(input_var) 
        print(output.shape)

if __name__ == '__main__':
    benchmark_name = sys.argv[1]
    if benchmark_name == "mel_scale":
        run_dmx_ops((32, 1024, 768), "mel_scale")
    elif benchmark_name == "reshape_casting":
        run_dmx_ops((1024, 16, 256), "reshape_casting")
    elif benchmark_name== "image_resize":
        run_dmx_ops((32, 3, 1024, 768), "image_resize") 