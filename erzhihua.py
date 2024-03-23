#!/usr/bin/env python
# encoding=gbk

'''
Canny��Ե��⣺�Ż��ĳ���
'''
import os

import cv2
import numpy as np
import torch
import utils


def CannyThreshold(lowThreshold, img):
    img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ת����ɫͼ��Ϊ�Ҷ�ͼ
    ratio = 3
    kernel_size = 3
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # ��˹�˲�
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)  # ��Ե���

    # just add some colours to edges from original image.
    # dst = cv2.bitwise_and(img,img,mask = detected_edges)  #��ԭʼ��ɫ��ӵ����ı�Ե��
    dst = detected_edges
    # dst1=255-dst
    dst1 = dst
    cv2.imshow('canny demo', dst1)
    return dst1
    # dst2=255-dst
    # cv2.imshow('canny demo',dst1)
    # cv2.imshow('canny demo2', dst2)



def get_edge(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = 3
        lowThreshold = 80
        kernel_size = 3
        detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # ��˹�˲�
        detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio,
                                   apertureSize=kernel_size)  # ��Ե���
        dst = detected_edges
        dst1 = torch.from_numpy(dst.astype(np.float32) / 255.0).contiguous().cuda()
        dst2 = dst1.unsqueeze(0)
        return dst2
def save_sample_png(sample_folder, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        print(img)
        # Process img_copy and do not destroy the data of img
        # img_copy = img.clone().data.permute( 1, 2 , 0).cpu().numpy()
        img_copy = img.clone().data.cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy2=img_copy
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = name_list[i] + '.jpg'
        save_img_path = os.path.join(sample_folder, save_img_name)
        # cv2.imwrite(save_img_path, img_copy)
        cv2.imwrite(save_img_path, img_copy2)

# img1 = cv2.imread("./test1/result/2_mask.jpg")
img1 = cv2.imread("./test3/mask1/1.png")
img1=torch.from_numpy(img1.astype(np.float32) / 255.0).contiguous().cuda()
img2=cv2.imread("./test3/edge1/1.jpg")
img2=torch.from_numpy(img2.astype(np.float32) / 255.0).contiguous().cuda()
threshold = 0.5  # ���ö�ֵ����ֵ��������Ҫ����
binary_img2 = (img2 > threshold).float()  # ������ֵ��������Ϊ1������Ϊ0

# �����Ҫ����ֵ��������ת��NumPy���飬����ʹ�����´��룺
img2 = binary_img2
print(img1)
# img3=cv2.imread("./test1/data/40000071.jpg")
# img3=torch.from_numpy(img3.astype(np.float32) / 255.0).contiguous().cuda()
# img4=img2 * (img1)+img3*(1-img1)
img3= (1-img1)*img2+img1
print(img3)
# img3=img2 * (img1)

img_list = [img3]
name_list = ['sample']
save_sample_png(sample_folder="./test1/sample/",  img_list=img_list, name_list=name_list,
                      pixel_max_cnt=255)


'''
�����ǵڶ���������cv2.createTrackbar()
����5����������ʵ������������������ʹ����֪����ʲô��˼��
��һ�������������trackbar���������
�ڶ��������������trackbar����������������
�����������������trackbar��Ĭ��ֵ,Ҳ�ǵ��ڵĶ���
���ĸ������������trackbar�ϵ��ڵķ�Χ(0~count)
������������ǵ���trackbarʱ���õĻص�������
'''
