# mural-inpatient

五台山壁画图像修复

##现有壁画图像修复方法存在问题：
1、无法根据专家指导生成相应结构
2、生成的纹理与已知区域纹理有视觉差距
## 模型结构图

![模型结构图](./image/1.png)

## SCA模块

![SCA模块图](./image/2.png)

## SCAP传播算法

![SCAp传播](./image/3.png)


## 实验结果
### 模型对比
![结果图](./image/4.png)
### 草图引导对比
![结果图](./image/5.png)
### 消融
####有无lossedge对比实验
![结果图](./image/6.png)
####有无SCA对比实验
![结果图](./image/7.png)
####有无SCAP对比实验
![结果图](./image/8.png)
### 破损壁画图像对比
![结果图](./image/9.png)

