import torch
from os import *
import time
import datetime
import numpy as np
from cv2 import*
import torch.cuda
import torch.nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from canny_track import CannyThreshold

from network import *
import train_dataset
import utils

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

# pthfile = r'./pretrained_model/deepfillv2_WGAN_D_epoch2100_batchsize2.pth'  # .pth文件的路径
pthfile = r'./models/deepfillv2_WGAN_G_epoch5001_batchsize2.pth'
model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  # 查看模型字典里面的key
    print(k)
# print('value:')
# for k in model:  # 查看模型字典里面的value
#     print(k, model[k])
# def w1(opt):
#   generator = util.create_generator(opt)
#   dict = torch.load('./pretrained_model/deepfillv2_WGAN_G_epoch2100_batchsize2.pth')
#   generator.load_state_dict(dict)
#   for name,param in generator.named_parameters():
#       print(name,param)
# #  #按参数名修改权重
#   b = torch.zeros((48, 5, 5, 5))
#   a = dict["coarse.0.conv2d.weight"]
#   for i in range(4):
#       b[:, i, :, :] = a[:, i, :, :]
#   torch.save(dict, './pretrained_model/model_0_.pth')
#   #验证修改是否成功
#   generator.load_state_dict(torch.load('./ckpt_dir//model_0_.pth'))
#   for param_tensor in generator.state_dict():
#       print(generator.state_dict()[param_tensor])