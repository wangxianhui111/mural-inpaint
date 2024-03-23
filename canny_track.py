#!/usr/bin/env python
# encoding=gbk
 
'''
Canny边缘检测：优化的程序
'''
import os

import cv2
import numpy as np
import torch
import utils


def CannyThreshold(lowThreshold,img):

    img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
    ratio = 3
    kernel_size = 3
    detected_edges = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波
    detected_edges = cv2.Canny(detected_edges,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测

     # just add some colours to edges from original image.
    # dst = cv2.bitwise_and(img,img,mask = detected_edges)  #用原始颜色添加到检测的边缘上
    dst=detected_edges
    # dst1=255-dst
    dst1=dst
    cv2.imshow('canny demo',dst1)
    return dst1
    # dst2=255-dst
    # cv2.imshow('canny demo',dst1)
    # cv2.imshow('canny demo2', dst2)

#
# def get_edge(img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ratio = 3
#         lowThreshold = 80
#         kernel_size = 3
#         detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
#         detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio,
#                                    apertureSize=kernel_size)  # 边缘检测
#         dst = detected_edges
#         dst1 = torch.from_numpy(dst.astype(np.float32) / 255.0).contiguous().cuda()
#         dst2 = dst1.unsqueeze(0)
#         return dst2
# def save_sample_png(sample_folder, img_list, name_list, pixel_max_cnt = 255):
#     # Save image one-by-one
#     for i in range(len(img_list)):
#         img = img_list[i]
#         # Recover normalization: * 255 because last layer is sigmoid activated
#         img = img * 255
#         # Process img_copy and do not destroy the data of img
#         img_copy = img.clone().data.permute( 1, 2 , 0).cpu().numpy()
#         img_copy = np.clip(img_copy, 0, pixel_max_cnt)
#         img_copy = img_copy.astype(np.uint8)
#         img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
#         # Save to certain path
#         save_img_name = name_list[i] + '.jpg'
#         save_img_path = os.path.join(sample_folder, save_img_name)
#         cv2.imwrite(save_img_path, img_copy)
#
# img = cv2.imread("D:/soft/21.jpg")
# img2=get_edge(img)
# img3=sketch=torch.cat((img2, img2, img2), 0)
# img_list = [img3]
# name_list = ['sketch']
# save_sample_png(sample_folder="D:/soft/2",  img_list=img_list, name_list=name_list,
#                       pixel_max_cnt=255)



'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
