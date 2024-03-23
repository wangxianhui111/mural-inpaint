import os

import numpy as np
import torch
import cv2
def save_sample_png(sample_folder, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
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
# 加载边缘检测图像
# edge_map = cv2.imread("D:/code/DeepFillv2_Pytorch-masterloss/samples/epoch112105_sketch.jpg", cv2.IMREAD_GRAYSCALE)  # 替换为实际的边缘检测图像文件地址，并指定加载为灰度图像
# # edge_map = cv2.imread("./test1/2.jpg", cv2.IMREAD_GRAYSCALE)
# # 将图像转换为Tensor
# edge_map = torch.tensor(edge_map).unsqueeze(0).unsqueeze(0).float()
# # 下采样倍率
# downscale_factor = 2
# target_height = edge_map.shape[2] // downscale_factor
# target_width = edge_map.shape[3] // downscale_factor
# # 计算下采样后的目标尺寸
# for _ in range(2):
#     # 使用双线性插值进行下采样
#     downsampled_edge_map = torch.nn.functional.interpolate(edge_map, size=(target_height, target_width),
#                                                            mode='bilinear', align_corners=False)
#     # 将下采样后的图像保存到指定路径
#     downsampled_edge_map = downsampled_edge_map.squeeze().clamp(0,
#                                                                 255).byte().numpy()  # 转换为numpy数组并去除冗余维度，并将像素值限制在0-255之间
#     threshold_value = 30  # 设定阈值
#     ret, downsampled_edge_map = cv2.threshold(downsampled_edge_map, threshold_value, 255, cv2.THRESH_BINARY)
#     downsampled_edge_map = torch.from_numpy(downsampled_edge_map.astype(np.float32) / 255.0).contiguous().cuda()
#     target_height = target_height // downscale_factor
#     target_width = target_width // downscale_factor



# edge_map_path = "D:/code/DeepFillv2_Pytorch-masterloss/samples/epoch112105_sketch.jpg"  # 边缘检测图像路径
# edge_map_path = "./test1/3.jpg"
# edge_map = cv2.imread(edge_map_path)
# print(type(edge_map))
edge_map = torch.randn(256, 256,3,3)
edge_map = edge_map.cpu().numpy()
# 连续进行三次双线性插值下采样
downscale_factor = 2

for _ in range(2):
    # 计算下采样后的目标尺寸
    target_height = edge_map.shape[0] // downscale_factor
    target_width = edge_map.shape[1] // downscale_factor

    # 使用双线性插值进行下采样
    edge_map = cv2.resize(edge_map, (target_width,target_height), interpolation=cv2.INTER_LINEAR)

    threshold_value = 55  # 设定阈值
    ret, edge_map = cv2.threshold(edge_map, threshold_value, 255, cv2.THRESH_BINARY)
edge_map = torch.from_numpy(edge_map.astype(np.float32) / 255.0).contiguous().cuda()
img_list = [edge_map]
name_list = ['sample']
save_sample_png(sample_folder="./test1/sample/",  img_list=img_list, name_list=name_list,
                      pixel_max_cnt=255)