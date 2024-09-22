import os
import cv2
import numpy as np
import json
from tqdm import tqdm


def labelme2mask_single_img(img_path, labelme_json_path):
    '''
    输入原始图像路径和labelme标注路径，输出 mask
    '''
    
    img_bgr = cv2.imread(img_path)
    img_mask = np.zeros(img_bgr.shape[:2]) # 创建空白图像 0-背景
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
        
    for one_class in class_info: # 按顺序遍历每一个类别
        for each in labelme['shapes']: # 遍历所有标注，找到属于当前类别的标注
            if each['label'] == one_class['label']:
                if one_class['type'] == 'polygon': # polygon 多段线标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（闭合区域）
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip': # line 或者 linestrip 线段标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（非闭合区域）
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'], thickness=one_class['thickness']) 

                elif one_class['type'] == 'circle': # circle 圆形标注

                    points = np.array(each['points'], dtype=np.int32)

                    center_x, center_y = points[0][0], points[0][1] # 圆心点坐标

                    edge_x, edge_y = points[1][0], points[1][1]     # 圆周点坐标

                    radius = np.linalg.norm(np.array([center_x, center_y] - np.array([edge_x, edge_y]))).astype('uint32') # 半径

                    img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'], one_class['thickness'])

                else:
                    print('未知标注类型', one_class['type'])
                    
    return img_mask

if __name__ == "__main__":
    img_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/eagleford'
    js_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/labels'
    mask_path = '/mnt/c/VScode/WS-Hub/WS-label2mask/img_masks'

    # 0-背景，从 1 开始
    class_info = [
        {'label':'Organic matter', 'type':'polygon', 'color':1},                    # polygon 多段线
        {'label':'Organic pore', 'type':'polygon', 'color':2},
        {'label':'Inorganic pore', 'type':'polygon', 'color':3},
        # {'label':'tower','type':'polygon','color':4},
        # {'label':'bus','type':'polygon','color':5},
        # {'label':'car','type':'polygon','color':6},
        # {'label':'tree','type':'polygon','color':7},
        # {'label':'fence','type':'polygon','color':8},
        # {'label':'wall','type':'polygon','color':9},
        # {'label':'person','type':'polygon','color':10},
        # {'label':'clock', 'type':'circle', 'color':11, 'thickness':-1},   # circle 圆形，-1表示填充
        # {'label':'lane', 'type':'line', 'color':12, 'thickness':5},       # line 两点线段，填充线宽
        # {'label':'sign', 'type':'linestrip', 'color':13, 'thickness':3}   # linestrip 多段线，填充线宽
]

    img_list = os.listdir(img_path)
    # js_list = os.listdir(js_path)
    
    # for img, js in zip(img_list, js_list):
    #     img_mask = draw_annotation(img_path, img, os.path.join(js_path, js), class_info)
    #     cv2.imwrite(os.path.join(mask_path, img[:-4] + '.png'), img_mask)
    
    for img in tqdm(img_list):
        
        try:
            labelme_json_path = os.path.join(js_path, '.'.join(img.split('.')[:-1])+'.json')
            img_mask = labelme2mask_single_img(os.path.join(img_path,img), labelme_json_path)
            mask_name = img.split('.')[0] + '.png'
            cv2.imwrite(os.path.join(mask_path, mask_name), img_mask)
        
        except Exception as E:
            print(img_path, '转换失败', E)