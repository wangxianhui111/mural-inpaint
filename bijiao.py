from tkinter import Image

import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compare_ssim

import os
from PIL import Image

# img = Image.open('./bijiao/cao1.jpg').convert('RGB')
# img.save('./bijiao/Scao1.jpg')

# img1 = cv2.imread('./bijiao/g1.jpg')
# img2 = cv2.imread('./bijiao/g2.jpg')
img3 = cv2.imread('/bijiao/S1.jpg')

img5 = cv2.imread('/bijiao/Scao1.jpg')
# print(img1.shape)
# print(img2.shape)
print(img5)
print(img3)



from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compare_ssim

MSE = mean_squared_error(img1, img2)
PSNR = peak_signal_noise_ratio(img1, img2)
SSIM =compare_ssim(img1,img2)

print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM',SSIM)