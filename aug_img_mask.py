import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
from tqdm import tqdm

# 随机翻转
def random_flip(image, mask):
    """
    Randomly flips the image and mask horizontally or vertically.
    """
    if np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return image, mask

# 高斯
def add_gaussian_noise(image):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(0, 25, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# 椒盐
def add_salt_and_pepper_noise(image):
    """
    Adds salt and pepper noise to the image.
    """
    noise = np.random.choice([-100, 100], size=image.shape, p=[0.5, 0.5])
    image = image + noise
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# 随机擦除
def random_erase(image, mask, p=0.25, scale=(0.02, 0.33), ratio=(0.5, 2.0), value=0):
    """
    Randomly erases a region of the image and mask.
    """
    if np.random.rand() < p:
        image = np.array(image)
        mask = np.array(mask)
        img_h, img_w = image.shape
        aspect_ratio = np.random.uniform(ratio[0], ratio[1])
        area = np.random.uniform(scale[0], scale[1]) * img_h * img_w
        h = int(np.sqrt(area / aspect_ratio))
        w = int(aspect_ratio * h)
        x = np.random.randint(0, img_w - w)
        y = np.random.randint(0, img_h - h)
        image[y:y+h, x:x+w] = value
        mask[y:y+h, x:x+w] = value
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
    return image, mask

# 透射变换
def perspective_transform(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    img_h, img_w = image.shape
    tl_x = np.random.randint(img_w//8, img_w//8*7)
    tl_y = np.random.randint(img_h//8, img_h//8*7)
    tr_x = np.random.randint(img_w//8*7, img_w)
    tr_y = np.random.randint(img_h//8, img_h//8*7)
    br_x = np.random.randint(img_w//8*7, img_w)
    br_y = np.random.randint(img_h//8*7, img_h)
    bl_x = np.random.randint(img_w//8, img_w//8*7)
    bl_y = np.random.randint(img_h//8*7, img_h)

    src = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
    dst = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, M, (img_w, img_h))
    mask = cv2.warpPerspective(mask, M, (img_w, img_h))
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    return image, mask

# 随机伸缩
def random_stretch(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    img_h, img_w = image.shape
    src_tri = np.array([[0, 0], [img_w, 0], [0, img_h]], dtype=np.float32)
    dst_tri = np.array([[0, 0], [img_w, 0], [np.random.uniform(img_w*0.45, img_w*0.55), img_h]], dtype=np.float32)

    M = cv2.getAffineTransform(src_tri, dst_tri)
    image = cv2.warpAffine(image, M, (img_w, img_h))
    mask = cv2.warpAffine(mask, M, (img_w, img_h))
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    return image, mask

#  旋转
def random_rotate(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    if np.random.rand() < 0.15:
        angle = np.random.randint(-15, 16)
    else:
        angle = np.random.randint(0, 4) * 90   # 旋转角度
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    return image, mask

def main():
    # 设定数据增强的次数
    num_augmentations = 30

    # 装载图片和mask掩码
    image_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/now/img_output'
    mask_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/now/mask_output'
    img_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    img_list = tqdm(img_list, desc="正在进行数据增强ing：")
    for im, ma in zip(img_list, mask_list):
        image = Image.open(os.path.join(image_path,im))
        mask = Image.open(os.path.join(mask_path,ma))

        # 创建输出文件夹
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im_aug_save_path = os.path.join(output_dir, 'img')
        if not os.path.exists(im_aug_save_path):
            os.makedirs(im_aug_save_path)

        ma_aug_save_path = os.path.join(output_dir, 'mask')
        if not os.path.exists(ma_aug_save_path):
            os.makedirs(ma_aug_save_path)

        # 进行数据增强
        for i in range(num_augmentations):
            image_aug, mask_aug = random_flip(image, mask)
            image_aug = np.array(image_aug)

            # 添加高斯噪声
            if np.random.rand() < 0.5:
                image_aug = add_gaussian_noise(image_aug)

            # if np.random.rand() < 0.5:
            #     image_aug = add_salt_and_pepper_noise(image_aug)

            # 随机擦除
            image_aug, mask_aug = random_erase(Image.fromarray(image_aug), mask_aug)

            # 随机旋转
            if np.random.rand() < 0.5:
                image_aug, mask_aug = random_rotate(image_aug, mask_aug)

            # # 伸缩变形
            # image_aug, mask_aug = random_stretch(image_aug, mask_aug)

            # # 透射变换
            # image_aug, mask_aug = perspective_transform(image_aug, mask_aug)

            # 保存增强后的图片和mask
            image_aug.save(os.path.join(im_aug_save_path, f'{im.split(".")[0]}_{i}.jpg'))
            mask_aug.save(os.path.join(ma_aug_save_path, f'{ma.split(".")[0]}_{i}.png'))

if __name__ == "__main__":
    main()
