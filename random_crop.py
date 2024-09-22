import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def random_crop(image, mask, crop_size):
    """
    Randomly crops the image and mask to the specified size.
    """
    w, h = image.size
    x = np.random.randint(0, w - crop_size)
    y = np.random.randint(0, h - crop_size)
    image_crop = image.crop((x, y, x + crop_size, y + crop_size))
    mask_crop = mask.crop((x, y, x + crop_size, y + crop_size))
    return image_crop, mask_crop

def main():
    # 设定裁剪的小图片张数
    num_crops = 75

    # 设定裁剪的大小
    crop_size = 320

    # 装载图片和mask掩码
    image_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/eagleford'
    mask_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/img_masks'
    img_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)

    for im, ma in zip(img_list, mask_list):
        image = Image.open(os.path.join(image_path, im))
        mask = Image.open(os.path.join(mask_path, ma))

        # 创建输出文件夹
        img_output_dir = 'img_output'
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
        mask_output_dir = 'mask_output'
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)

        # 进行随机裁剪
        for i in range(num_crops):
            image_crop, mask_crop = random_crop(image, mask, crop_size)
            image_crop.save(os.path.join(img_output_dir, f'{im.split('.')[0]}_{i}.jpg'))
            mask_crop.save(os.path.join(mask_output_dir, f'{ma.split('.')[0]}_{i}.png'))

if __name__ == "__main__":
    main()
