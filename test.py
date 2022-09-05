import torch
import torch.nn as  nn

# value = torch.randn(12,196,384)
# B, N, C = value.shape
# qkv= nn.Linear(384,384,bias=False)
# print(qkv(value).shape)
# ss = qkv(value).reshape(B, N, 12, C // 12).permute(0, 2, 1, 3)
# print(ss.shape)



#计算参数量

class depth(nn.Module):

    def __init__(self):
        super(depth, self).__init__()
        self.kernel= 7
        self.stride = 4
        self.image = torch.randn(12,3,224,224)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=self.kernel,stride=self.stride,padding=2)
        self.conv2 = nn.Conv2d(3,48,kernel_size=4,stride=self.stride/2,padding=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=4, stride=self.stride / 2, padding=1)

    def forward(self,):
        value = self.conv1(self.image)
        return value

    def convolutionTwo(self):  
        value = self.conv3(self.conv2(self.image))
        return value

from ptflops import get_model_complexity_info
model =depth()
flops,params =get_model_complexity_info(model,input_res=(3,224,224),as_strings=True,print_per_layer_stat=True)
print(f'flops :{flops}'+'\t'+f'params:{params}')