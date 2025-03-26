import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance

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
    if isinstance(image, Image.Image):
        image = np.array(image)
    noise = np.random.normal(0, 25, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# # 椒盐
# def add_salt_and_pepper_noise(image):
#     """
#     Adds salt and pepper noise to the image.
#     """
#     noise = np.random.choice([-100, 100], size=image.shape, p=[0.5, 0.5])
#     image = image + noise
#     image = np.clip(image, 0, 255)
#     return image.astype(np.uint8)

# 随机擦除
def random_erase(image, mask, p=0.3, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0):
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
# def perspective_transform(image, mask):
#     image = np.array(image)
#     mask = np.array(mask)
#     img_h, img_w = image.shape
#     tl_x = np.random.randint(img_w//8, img_w//8*7)
#     tl_y = np.random.randint(img_h//8, img_h//8*7)
#     tr_x = np.random.randint(img_w//8*7, img_w)
#     tr_y = np.random.randint(img_h//8, img_h//8*7)
#     br_x = np.random.randint(img_w//8*7, img_w)
#     br_y = np.random.randint(img_h//8*7, img_h)
#     bl_x = np.random.randint(img_w//8, img_w//8*7)
#     bl_y = np.random.randint(img_h//8*7, img_h)

#     src = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
#     dst = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src, dst)
#     image = cv2.warpPerspective(image, M, (img_w, img_h))
#     mask = cv2.warpPerspective(mask, M, (img_w, img_h))
#     image = Image.fromarray(image)
#     mask = Image.fromarray(mask)
#     return image, mask

# # 随机伸缩
# def random_stretch(image, mask):
#     image = np.array(image)
#     mask = np.array(mask)
#     img_h, img_w = image.shape
#     src_tri = np.array([[0, 0], [img_w, 0], [0, img_h]], dtype=np.float32)
#     dst_tri = np.array([[0, 0], [img_w, 0], [np.random.uniform(img_w*0.45, img_w*0.55), img_h]], dtype=np.float32)

#     M = cv2.getAffineTransform(src_tri, dst_tri)
#     image = cv2.warpAffine(image, M, (img_w, img_h))
#     mask = cv2.warpAffine(mask, M, (img_w, img_h))
#     image = Image.fromarray(image)
#     mask = Image.fromarray(mask)
#     return image, mask

#  旋转
def random_rotate(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    if np.random.rand() < 0.6:
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

# 对比度增强
def adjust_contrast(image, alpha=None):
    """
    Adjusts the contrast of an image.
    alpha > 1 increases contrast.
    alpha < 1 decreases contrast.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if np.random.rand()> 0.5:
        alpha = 1 + np.random.uniform(0, 0.4)
    else:
        alpha = 1 - np.random.uniform(0, 0.4)
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(alpha)
    return image_enhanced

# 亮度调整
def adjust_brightness(image, beta=None):
    """
    Adjusts the brightness of an image.
    beta > 1 increases brightness.
    beta < 1 decreases brightness.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if np.random.rand()> 0.5:
        beta = 1 + np.random.uniform(0, 0.2)
    else:
        beta = 1 - np.random.uniform(0, 0.2)
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(beta)
    return image_enhanced

# 锐化
def sharpen(image, sigma=1.5):
    """
    Sharpens an image using an unsharp mask.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    factor = 1 + sigma
    arr = np.array(image)
    if len(arr.shape) == 3:  # color image
        sharpened = np.array([np.clip((arr[..., i] * factor - 0.5 * (arr[..., i-1] + arr[..., i+1])).astype(arr.dtype), 0, 255) for i in range(1, arr.shape[2]-1)])
    else:  # grayscale image
        sharpened = np.clip((arr * factor - 0.5 * (np.roll(arr, 1, axis=0) + np.roll(arr, 1, axis=1))).astype(arr.dtype), 0, 255)
    image_sharpened = Image.fromarray(sharpened)
    return image_sharpened

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def unsharp_mask(image, sigma=1.5, strength=0.8):
    """
    Apply unsharp masking to an image.
    :param image: Input image (PIL Image or numpy array)
    :param sigma: Standard deviation for Gaussian blur
    :param strength: Strength of sharpening
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to numpy array
    arr = np.array(image).astype(np.float32)
    
    # Apply Gaussian blur
    blurred = gaussian_filter(arr, sigma=sigma)
    
    # Calculate the mask
    mask = arr - blurred
    
    # Apply the mask to the original image
    sharpened = np.clip(arr + strength * mask, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    sharpened_image = Image.fromarray(sharpened)
    return sharpened_image



def main():
    # 设定数据增强的次数
    num_augmentations = 50

    # 装载图片和mask掩码
    image_path = '/mnt/e/VScode/WS-Hub/WS-label2mask/img_output_changed_256'
    mask_path = '/mnt/e/VScode/WS-Hub/WS-label2mask/mask_output_changed_256'
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

        im_aug_save_path = os.path.join(output_dir, 'chged_img_256_crop_80')
        if not os.path.exists(im_aug_save_path):
            os.makedirs(im_aug_save_path)

        ma_aug_save_path = os.path.join(output_dir, 'chged_mask_256_crop_80')
        if not os.path.exists(ma_aug_save_path):
            os.makedirs(ma_aug_save_path)

        # 进行数据增强
        for i in range(num_augmentations):
            image_aug, mask_aug = random_flip(image, mask)
            image_aug = np.array(image_aug)
            
            # 锐化 (锐化必须在加噪之前)
            if np.random.rand() < 0.1:
                image_aug = sharpen(image_aug)
                
            # 添加高斯噪声
            if np.random.rand() < 0.5:
                image_aug = add_gaussian_noise(image_aug)

            # if np.random.rand() < 0.5:
            #     image_aug = add_salt_and_pepper_noise(image_aug)
            
            # 对比度增强
            if np.random.rand() < 0.5:
                image_aug = adjust_contrast(image_aug)

            # 亮度调整
            if np.random.rand() < 0.35:
                image_aug = adjust_brightness(image_aug)
            
            
            # 随机擦除
            image_aug, mask_aug = random_erase(image_aug, mask_aug)

            # 随机旋转
            if np.random.rand() < 0.5:
                image_aug, mask_aug = random_rotate(image_aug, mask_aug)


            # 保存增强后的图片和mask
            if isinstance(image_aug, np.ndarray):
                image_aug = Image.fromarray(image_aug)
            image_aug.save(os.path.join(im_aug_save_path, f'{im.split(".")[0]}_{i}.jpg'))
            mask_aug.save(os.path.join(ma_aug_save_path, f'{ma.split(".")[0]}_{i}.png'))

if __name__ == "__main__":
    main()
