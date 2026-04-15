import torch
import torch.nn as nn
import torch.nn.functional as F
import math
   
def spatial_pyramid_pool(previous_conv, levels=[4, 2, 1]):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    # exit()
    c,h,w = previous_conv.size()
    num_sample = 1
    previous_conv_size = (h, w)
    out_pool_size = [4, 2, 1]
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_str = int(math.floor(previous_conv_size[0] / out_pool_size[i]))
        w_str = int(math.floor(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2

        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_str, w_str), padding=(0))
        x = maxpool(previous_conv)
        # print("prev_conv_size", i, previous_conv.size())
        # print("x", i, x.size())
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), dim=1)
        # print(spp.size())
        # exit()
    return spp

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=674, hidden_dim=128, mid_dim=64, num_classes=6):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, 512)
        self.fc2  = nn.Linear(512, 256)
        self.fc3  = nn.Linear(256, 128)
        self.fc4  = nn.Linear(128, 64)
        self.head = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(self.head(x))          