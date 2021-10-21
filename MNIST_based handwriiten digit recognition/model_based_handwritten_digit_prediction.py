"""
读取图片，转灰度，resize到28
传入mnist模型中predict
"""

from __future__ import division, print_function, absolute_import

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # 卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
        # 全连接层
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10))
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)  # 将参数扁平化，否则输入至全连接层会报错
        x = self.dense(x)
        return x   
    
model = torch.load(r'C:\Users\未央\Desktop\rr\mnist.pkl',map_location=torch.device('cpu'))
model = model.double()

# 读取图片转成灰度格式
img = Image.open(r'C:\Users\未央\Desktop\2.bmp').convert('L')

# resize的过程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

# 暂存像素值的一维数组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = 1.0 - float(img.getpixel((j, i)))/255.0
        # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 1, 28, 28))

plt.imshow(arr1[0,0,:,:])

arr1 = torch.tensor(arr1)

print(arr1.shape)

# 获取数据并处理
outputs = model(arr1)

_,pred = torch.max(outputs.data,1)

print(pred)