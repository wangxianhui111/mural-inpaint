
import torch

# 加载权重文件
weights1 = torch.load('./models/deepfillv2_WGAN_G_epoch40000400_batchsize4.pth')
weights2 = torch.load('./models/deepfillv2_WGAN_G_epoch40000500_batchsize4.pth')

# 创建新的权重字典
weights3 = {}

# 复制权重文件2的所有内容到新字典
for key, value in weights2.items():
    weights3[key] = value

# 更新含有"coarse"关键字的value
for key, value in weights1.items():
    if 'coarse' in key:
        weights3[key] = value

# 保存新的权重文件
torch.save(weights3, './4/3.pth')
