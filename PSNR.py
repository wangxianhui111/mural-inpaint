import os
import cv2
import numpy as np
import skimage
import torch
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    imdff = np.clip(img2, 0, 1) - np.clip(img1, 0, 1)
    rmse = np.sqrt(np.mean(imdff ** 2))
    ps = 20 * np.log10(1 / rmse)
    return ps

def calculate_ssim(img1, img2):
    img1=torch.from_numpy(img1)
    img2=torch.from_numpy(img2)
    return ssim(img1, img2)

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def save_image_with_metrics(img, img_name, psnr, ssim, mae):
    img_name_parts = img_name.split('.')
    img_name_parts[-2] += f'_PSNR{psnr:.2f}_SSIM{ssim:.4f}_MAE{mae:.4f}'
    new_img_name = '.'.join(img_name_parts)
    cv2.imwrite(os.path.join('3', new_img_name), img)

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1).cuda() - torch.clamp(tar_img, 0, 1).cuda()
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    ps = ps.cpu()
    return ps.item()

def ssim(pred, target):
    pred = pred.clone().data.cpu().numpy()
    target = target.clone().data.cpu().numpy()
    target = target[0]
    pred = pred[0]
    # ssim = skimage.measure.compare_ssim(target, pred)
    ssim = skimage.metrics.structural_similarity(target,pred,
                           win_size=None, gradient=False, data_range=None,
                          multichannel=True, gaussian_weights=False,
                          full=False)
    return ssim
def process_images(image_dir1, image_dir2):
    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    image_count = 0

    image_files1 = sorted(os.listdir(image_dir1), key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(image_files1)
    image_files2 = sorted(os.listdir(image_dir2), key=lambda x: int(''.join(filter(str.isdigit, x))))  # 对文件列表进行排序
    print(image_files2)

    for img_name1, img_name2 in zip(image_files1, image_files2):
        img_path1 = os.path.join(image_dir1, img_name1)
        img_path2 = os.path.join(image_dir2, img_name2)

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        img3 = img2
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if img1 is None:
            print(f"无法加载图像：{img_path1}")
            continue
        if img2 is None:
            print(f"无法加载图像：{img_path2}")
            continue
        img1 = img1 / 255
        img2 = img2 / 255

        psnr = calculate_psnr(img1, img2)
        ssim_value = calculate_ssim(img1, img2)
        mae = calculate_mae(img1, img2)

        save_image_with_metrics(img3, img_name1, psnr, ssim_value, mae)

        total_psnr += psnr
        total_ssim += ssim_value
        total_mae += mae
        image_count += 1

    avg_psnr = total_psnr / image_count
    avg_ssim = total_ssim / image_count
    avg_mae = total_mae / image_count

    with open('1.txt', 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.2f}\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
        f.write(f'Average MAE: {avg_mae:.4f}\n')

image_dir1 = '2/'  # 生成图片所在目录
image_dir2 = '1/'  # 真实图片所在目录
process_images(image_dir1, image_dir2)
