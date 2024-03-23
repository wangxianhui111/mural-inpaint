from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models
import torchvision.transforms as transforms

# imgPath= "./results/1_second_out.jpg"
imgPath= "./test_data/5381.jpg"
model = models.vgg16(pretrained=True).features
model.eval()

input= transforms.ToTensor()(Image.open(imgPath)).unsqueeze(0)
print(input.shape) 
"""
outputs:
torch.Size([1, 3, 720, 1280])
"""

x= input
out= [x.squeeze(0).detach().numpy().transpose(2,1,0)[..., 0:3]]
for index,layer in enumerate(model):
    print(index, layer) # 得到上面vgg16结构
    x= layer(x)
    out.append(x.squeeze(0).detach().numpy().transpose(2,1,0)[...,0:3])
    # 打印所有池化层的输出
    if index in [4, 9, 16, 23, 30]:
        plt.imshow(x.squeeze(0).detach().numpy().transpose(2,1,0)[...,0:3])
        plt.show()

# 打印所有层的输出
plt.figure()
for i in range(1, 33):
    plt.subplot(4,8, i)
    plt.imshow(out[i-1])
    plt.xticks()
    plt.yticks()
plt.show()