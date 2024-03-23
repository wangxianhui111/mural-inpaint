import cv2
import numpy as np
import torch

def downsketch(sketch, x):
    downscale_factor = 2
    edge_map = sketch.cpu().numpy()

    for _ in range(x):
        # 计算下采样后的目标尺寸
        target_height = edge_map.shape[2] // downscale_factor
        target_width = edge_map.shape[3] // downscale_factor

        # 将图像数据转换为 uint8 类型
        edge_map = edge_map.astype(np.uint8)

        # 使用双线性插值进行下采样
        edge_map = cv2.resize(edge_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        threshold_value = 55  # 设定阈值
        _, edge_map = cv2.threshold(edge_map, threshold_value, 255, cv2.THRESH_BINARY)

    # 将结果转换为 Tensor，并移动回 GPU
    edge_map = torch.from_numpy(edge_map.astype(np.float32) / 255.0).contiguous().cuda()

    return edge_map

# 示例使用
sketch = torch.randn(1, 3, 256, 256)
sketch = sketch.cuda()
downsampled_sketch = downsketch(sketch, 3)
