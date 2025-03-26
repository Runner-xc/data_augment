import os
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm
import random
import hashlib


class ShaleCropper:
    def __init__(self, crop_size=512, min_crops=40, max_crops=60,
                 pore_classes=[2, 3], max_overlap=0.3):
        """
        强化版智能裁剪器（支持差异化采样）
        :param crop_size: 裁剪尺寸
        :param min_crops: 最小裁剪数
        :param max_crops: 最大裁剪数
        :param pore_classes: 关注的孔隙类别
        :param max_overlap: 最大允许重叠比例
        """
        self.crop_size = crop_size
        self.min_crops = min_crops
        self.max_crops = max_crops
        self.pore_classes = pore_classes
        self.max_overlap = max_overlap
        self.rng = random.Random()  # 独立随机数生成器

    def _calculate_porosity(self, mask_crop):
        """计算孔隙占比"""
        mask_array = np.array(mask_crop)
        pores = np.sum(np.isin(mask_array, self.pore_classes))
        classes = np.sum(np.isin(mask_array, [1, 2, 3]))
        return pores / classes if classes != 0 else 0

    def _grid_sampling(self, width, height):
        """网格化空间采样（增强随机扰动）"""
        grid_crops = []
        w_grid_division = max(1, width // (self.crop_size//2))
        h_grid_division = max(1, height // (self.crop_size//2))
        x_steps = np.linspace(0, max(0, width-self.crop_size), w_grid_division)
        y_steps = np.linspace(0, max(0, height-self.crop_size), h_grid_division)
        
        for x in x_steps:
            for y in y_steps:
                # 增大随机偏移范围至crop_size//2
                dx = self.rng.randint(0, self.crop_size//2)
                dy = self.rng.randint(0, self.crop_size//2)
                x0 = int(np.clip(x + dx, 0, width-self.crop_size))
                y0 = int(np.clip(y + dy, 0, height-self.crop_size))
                grid_crops.append((x0, y0, x0+self.crop_size, y0+self.crop_size))
        return grid_crops
    
    def _random_sampling(self, width, height, num_samples):
        """差异化随机采样"""
        random_samples = []
        for _ in range(num_samples):
            # 使用实例内部的随机生成器
            w = self.rng.randint(0, max(0, width-self.crop_size))
            h = self.rng.randint(0, max(0, height-self.crop_size))
            random_samples.append((w, h, w+self.crop_size, h+self.crop_size))
        return random_samples
    
    def _check_overlap(self, new_crop, existing_crops):
        """改进型重叠检查"""
        if not existing_crops:
            return False
        
        new_area = self.crop_size ** 2
        for crop in existing_crops:
            dx = min(new_crop[2], crop[2]) - max(new_crop[0], crop[0])
            dy = min(new_crop[3], crop[3]) - max(new_crop[1], crop[1])
            overlap = max(0, dx) * max(0, dy)
            if overlap / new_area > self.max_overlap:
                return True
        return False
    
    def smart_crop(self, image, mask):
        """改进型智能裁剪（增加旋转扰动）"""
        w, h = image.size
        if w < self.crop_size or h < self.crop_size:
            return []
        
        # 动态生成候选框
        grid_crops = self._grid_sampling(w, h)
        random_crops = self._random_sampling(w, h, self.max_crops*2)
        all_crops = grid_crops + random_crops
        
        # ROI采样增强（带随机旋转）
        mask_array = np.array(mask)
        prosity_mask = np.isin(mask_array, self.pore_classes)
        labeled = label(prosity_mask)
        for region in regionprops(labeled):
            y0, x0, y1, x1 = region.bbox
            center_x, center_y = (x0+x1)//2, (y0+y1)//2
            for _ in range(5):
                # 增加旋转扰动 (-15度到15度)
                angle = self.rng.uniform(-15, 15)
                # 使用旋转后的mask进行坐标变换
                rotated_mask = mask.rotate(angle, expand=False, fillcolor=0)
                rotated_array = np.array(rotated_mask)
                # 重新计算ROI区域
                if np.any(rotated_array[y0:y1, x0:x1]):
                    dx = self.rng.randint(-self.crop_size, self.crop_size)
                    dy = self.rng.randint(-self.crop_size, self.crop_size)
                    x_start = np.clip(center_x + dx - self.crop_size//2, 0, w-self.crop_size)
                    y_start = np.clip(center_y + dy - self.crop_size//2, 0, h-self.crop_size)
                    all_crops.append((x_start, y_start, x_start+self.crop_size, y_start+self.crop_size))

        # 候选框筛选流程
        valid_crops = []
        for crop in all_crops:
            # 边界有效性检查
            if (crop[2]-crop[0] != self.crop_size) or (crop[3]-crop[1] != self.crop_size):
                continue
                
            # 孔隙率阈值检查
            mask_crop = mask.crop(crop)
            if self._calculate_porosity(mask_crop) <= 0.001:
                continue
                
            # 重叠检测
            if not self._check_overlap(crop, valid_crops):
                valid_crops.append(crop)
            
            if len(valid_crops) >= self.max_crops * 2:  # 扩大候选池
                break

        # 差异化筛选策略
        final_crops = []
        seen = set()
        # 优先选择网格采样且唯一的区域
        for c in valid_crops:
            if c in grid_crops and c not in seen:
                final_crops.append(c)
                seen.add(c)
            if len(final_crops) >= self.min_crops:
                break
        
        # 补充随机样本
        remaining = min(self.max_crops, len(valid_crops)) - len(final_crops)
        for c in valid_crops:
            if c not in seen and remaining > 0:
                final_crops.append(c)
                seen.add(c)
                remaining -= 1
        
        # 最终随机裁剪数量
        return final_crops


def main():
    config = {
        "input": {
            "images": "/mnt/e/VScode/WS-Hub/WS-label2mask/eagleford",
            "masks": "/mnt/e/VScode/WS-Hub/WS-label2mask/img_masks"
        },
        "output": {
            "images": "./cropped_images",
            "masks": "./cropped_masks"
        },
        "crop_size": 256,
        "min_crops": 80,
        "max_crops": 120,
        "pore_classes": [2, 3],
        "max_overlap" : 0.6
    }

    cropper = ShaleCropper(
        crop_size=config["crop_size"],
        min_crops=config["min_crops"],
        max_crops=config["max_crops"],
        pore_classes=config["pore_classes"],
        max_overlap=config["max_overlap"]
    )

    # 准备输出目录
    output_images_path = os.path.join(config["output"]["images"], str(config['crop_size']))
    output_masks_path = os.path.join(config["output"]["masks"], str(config['crop_size']))
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)

    # 处理流程
    for img_file in tqdm(os.listdir(config["input"]["images"])):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base_name = os.path.splitext(img_file)[0]
        
        # 基于文件名生成种子
        unique_seed = int(hashlib.sha256(img_file.encode()).hexdigest(), 16) % (2**32)
        random.seed(unique_seed)
        np.random.seed(unique_seed)
        cropper.rng = random.Random(unique_seed)  # 重置实例随机状态
        
        # 加载数据
        img_path = os.path.join(config["input"]["images"], img_file)
        mask_path = os.path.join(config["input"]["masks"], f"{base_name}.png")
        

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 执行智能裁剪
        crops = cropper.smart_crop(image, mask)
        
        # 保存结果
        for i, (x0, y0, x1, y1) in enumerate(crops):
            img_crop = image.crop((x0, y0, x1, y1))
            mask_crop = mask.crop((x0, y0, x1, y1))
            
            img_crop.save(os.path.join(output_images_path, f"{base_name}_crop{i:03d}.jpg"))
            mask_crop.save(os.path.join(output_masks_path, f"{base_name}_crop{i:03d}.png"))


if __name__ == "__main__":
    main()
