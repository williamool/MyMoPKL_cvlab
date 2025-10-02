import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch
import torch.nn as nn
import pickle

# 转换为RGB图像
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
# 归一化
def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

# 生成随机数
def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a


class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5 ,type='train'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.type = type
        self.image_size = image_size
        self.num_frame = num_frame
        # 训练集开启数据增强
        if type == 'train':
            self.txt_path = dataset_path
            self.aug = True
        else:
            self.txt_path = dataset_path
            self.aug = False
        # 加载标注信息
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                # 转换Windows路径为Linux路径
                img_path = line[0].replace('D:/Github/ITSDT', '/home/wanboling/disk2/ITSDT')
                self.img_idx.append(img_path)
                self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))

        # 加载文本描述与运动关系
        ###############################
        description = pickle.load(open('/home/wanboling/disk2/MyMoPKL/emb_train_ITSDT.pkl', 'rb'))
        embeddings = np.array(list(description.values()))
        self.cap_idx =list(description.keys())
        self.motion_cap_idx = np.array(list(description.values()))

        relation = pickle.load(open('/home/wanboling/disk2/MyMoPKL/motion_relation_ITSDT.pkl', 'rb'))
        relations = np.array(list(relation.values()))
        self.re_idx = list(relation.keys())
        self.motion_re_idx = np.array(list(relation.values()))
        ###############################

    # 数据集长度
    def __len__(self):
        return self.length
    
    # 获取一个样本的完整信息
    def __getitem__(self, index):
        images, box, path = self.get_data(index) # 获取图像与标注
        images = np.transpose(preprocess(images),(3, 0, 1, 2)) # 图像归一化 调整维度顺序
        
        # 替换标准框形式为中心点+宽高
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        
        # 获取描述与框信息
        if self.type == 'train':
            caption_data,relation = self.get_caption_data(index)
            multi_box = self.get_boxes_data(index)
            for box in multi_box: 
                if len(box) != 0:
                    box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                    box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        else:
            caption_data = None
            multi_box = None
            relation = None
        # 转成numpy数组
        caption_data = np.array(caption_data)
        relation = np.array(relation)
       
        return images, box, caption_data, multi_box, relation, path 

    # 根据图片索引返回当前帧以及前几帧文本描述 返回当前帧运动描述
    def get_caption_data(self, index):
        file_name = self.img_idx[index]
        caption_frames = []
        relation = []
        
        # 尝试直接匹配完整路径
        try:
            cap_index = self.cap_idx.index(file_name)
        except ValueError:
            # 如果直接匹配失败，尝试匹配文件名
            file_name_only = file_name.split('/')[-1]  # 获取文件名部分
            try:
                cap_index = self.cap_idx.index(file_name_only)
            except ValueError:
                # 如果还是失败，尝试匹配相对路径
                relative_path = '/'.join(file_name.split('/')[-2:])  # 获取最后两级路径
                try:
                    cap_index = self.cap_idx.index(relative_path)
                except ValueError:
                    # 最后尝试：在cap_idx中查找包含该文件名的项
                    for i, cap_key in enumerate(self.cap_idx):
                        if file_name_only in cap_key or file_name in cap_key:
                            cap_index = i
                            break
                    else:
                        raise ValueError(f"无法找到匹配的caption数据: {file_name}")
        
        relation.append(self.motion_re_idx[cap_index])
        for id in range(0, self.num_frame):
            idx = max(cap_index - id, 0)
            caption_frames.append(self.motion_cap_idx[idx]) 
        return caption_frames[::-1], relation


    def get_boxes_data(self, index):
        # 获取图片信息
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        img = Image.open(image_path + '%d.bmp' % image_id)
        
        # 计算图片放缩和填充参数
        h, w = self.image_size, self.image_size
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
    
        # 获取标注信息
        boxes_frames = []
        with open(self.txt_path) as f:
            data_lines = f.readlines()

        # 采样多帧标注
        for id in range(self.num_frame):
            idx = max(index - id, 0)
            line = data_lines[idx].strip().split()
            if len(line) > 1:
                label_data = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            else:
                label_data = np.empty((0, 5))
            
            # 放缩标注并映射到新图尺寸上
            if label_data.size > 0:
                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy
            
                # 边界裁剪
                label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
                label_data[:, 2][label_data[:, 2] > w] = w
                label_data[:, 3][label_data[:, 3] > h] = h

            # 储存新标注信息
            boxes_frames.append(label_data)
        boxes_frames = [np.array(b, dtype=np.float32) for b in boxes_frames[::-1]]
        
        # 返回目标框数据 用于运动建模
        return boxes_frames

    
    def get_data(self, index):
        # 初始化
        image_data = []
        multi_frame_label = [] 
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        label_data = self.anno_idx[index]  # 4+1

        path = image_path +'%d.bmp' % image_id
        
        # 从当前帧往前读取num_frame帧
        for id in range(0, self.num_frame):
            
            img = Image.open(image_path +'%d.bmp' % max(image_id - id, 0))
            # 转换为RGB图像
            img = cvtColor(img)
            iw, ih = img.size
            
            # 计算缩放比例+padding
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            
            # 缩放+padding
            img = img.resize((nw, nh), Image.Resampling.BICUBIC)  
            new_img = Image.new('RGB', (w,h), (128, 128, 128)) 
            new_img.paste(img, (dx, dy)) 
            image_data.append(np.array(new_img, np.float32))
            
            # 处理标注框
            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]]*nw/iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]]*nh/ih + dy
                
                # 边界裁剪+筛去过小的框
                label_data[:, 0:2][label_data[:, 0:2]<0] = 0
                label_data[:, 2][label_data[:, 2]>w] = w
                label_data[:, 3][label_data[:, 3]>h] = h
               
                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w>1, box_h>1)] 

        # 返回帧序列与标注        
        image_data = np.array(image_data[::-1])
        label_data = np.array(label_data, dtype=np.float32)
        
        return image_data, label_data, path

# 接收并处理一个batch的数据           
def dataset_collate(batch):
    images = []
    bboxes = []
    captions = [] 
    multi_boxes = []
    relations = []
    paths = []
    for img, box, caption, multi_box, relation, path in batch:
        images.append(img)
        bboxes.append(box)
        captions.append(caption) 
        multi_boxes.append(multi_box) 
        relations.append(relation)
        paths.append(path)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    
    return images, bboxes, captions, multi_boxes, relations, paths
