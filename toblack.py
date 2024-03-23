# get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
import cv2
import numpy as np
import torch

# from canny_track import CannyThreshold
# for i in range(2):
#     img = cv2.imread("./testmask_sketch/1/%d.jpg"%(i+1),0)
    # print(img.shape)
    # print(img[:,:])
# img = cv2.imread("D:/soft/1.jpg",0)
# img = cv2.imread("D:/soft/1.jpg")

#转颜色
# def BGR_to_RGB(cvimg):
#     pilimg = cvimg.copy()
#     pilimg[:, :, 0] = cvimg[:, :, 2]
#     pilimg[:, :, 2] = cvimg[:, :, 0]
#     return pilimg
# img1 = cv2.imread("D:/soft/1.jpg")
# img2 = cv2.imread("D:/soft/2.jpg")
# white_pixels = np.where(
#           (img2[:,:, :] >= 127))
# black_pixels = np.where(
#           (img2[:,:, :] < 127))
# img2[black_pixels] = [0]
# img1[white_pixels]=[255]
# img2[white_pixels] = [255]
# img2=img2[:,:,1]
# img2=np.expand_dims(img2,axis=2)
# print(img1.shape)
# print(img2.shape)
# img3= img1 * (255-img2)+img2
# cv2.imshow('1',img3)
# img4=BGR_to_RGB(img3)
# cv2.imshow('2',img3)
# cv2.imwrite(r'D:/soft/1/100.jpg', img1)


# img= CannyThreshold(80,img)
    # black_pixels = np.where(
    #      (img[:, :, 0] != 0) &
    #      (img[:, :, 1] != 0) &
    # #      (img[:, :, 2] != 0)
    # #  )
    # black_pixels = np.where(
    #      (img[:, :] > 250)&(img[:, :] != 0)
    #   )
    # img[black_pixels] = [0]
    # cv2.imwrite(r'./testdata_sketch/%d.jpg'%(i+1), img)
# img = torch.from_numpy(img.astype(np.float32) / 255.0).contiguous()
# img = img[:,:,0]
# print(img.shape)
# count=0
# count2=0
# for i in range(0,256):
#     for j in range(0,256):
#         if img[i,j]==0:
#             count=count+1
#         # elif img[i,j]!=0:
#         #     count2=count2+1
#         else:
#             print(img[i,j])
# print(count)
# print(count2)



#
img = cv2.imread("D:/soft/11.jpg")
for i in range(1,11):
    cv2.imwrite(r'D:/soft/1/%d.jpg'%(i), img)








#
# black_pixels = np.where(
#     (img[:, :, 0] == 0) &
#     (img[:, :, 1] == 0) &
#     (img[:, :, 2] == 0)
# )
#
# # set those pixels to white
# img[black_pixels] = [255, 255, 255]
