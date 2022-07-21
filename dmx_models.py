from torch.nn import Module
from torch import nn
import torch


cnt = 0
def print_shape( x):
        global cnt
        print (f"the shape at {cnt} is {x.shape}")
        cnt += 1

# taken from benchmarks/model_generator.py: create_dmx_ops() function
class mel_scale(nn.Module):
    def __init__(self):
        super(mel_scale, self).__init__()
        
    def forward(self, x):
        xt = torch.transpose(x,1,2)
        y = torch.pow(xt,2)
        y = torch.mul(y,0.001)
        y = torch.add(y,1)
        y = torch.tanh(y) # replace log with tanh
        y = torch.mul(y,2595) 
        y = y.type(torch.CharTensor)
        return y

class reshape_casting(nn.Module):
    def __init__(self):
        super(reshape_casting, self).__init__()

    def forward(self, x):
        y = torch.pow(x,2)
        y = torch.mul(x,0.5) # normalization constant                                
        yt = torch.transpose(y,1,2)
        y = y.type(torch.CharTensor)        
        return yt

class image_resize(nn.Module):
    def __init__(self):
        super(image_resize, self).__init__()            
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        #y  = nn.Identity(x)
        xt = torch.transpose(x,1,2)           
        z = self.max_pool(xt)
        return z
