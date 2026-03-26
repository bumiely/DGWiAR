# coding=utf-8
import random

import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_train(dataset, resize_size=256, crop_size=224):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if dataset == "CSI":
        return transforms.Compose([
            # RandomFixedSizeBlocker(block_width_percentage=0.1, p=0.5),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(dataset, resize_size=256, crop_size=224):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if dataset == "CSI":
        return transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class RandomFixedSizeBlocker:
    def __init__(self, block_width_percentage=0.1, p=0.5):
        self.block_width_percentage = block_width_percentage  # 遮挡矩形宽度是图像宽度的百分比
        self.p = p  # 遮挡的概率

    def __call__(self, image):
        if random.random() < self.p:  # 根据概率决定是否进行遮挡
            width, height = image.size
            block_width = int(width * self.block_width_percentage)  # 计算遮挡矩形的宽度
            block_height = height  # 遮挡矩形的高度与图像高度相同

            # 随机选择遮挡区域的水平位置
            left = random.randint(0, width - block_width)
            top = 0  # 从图像顶部开始，遮挡区域高度覆盖整个图像
            right = left + block_width
            bottom = top + block_height

            # 创建一个绘制对象，填充遮挡区域
            draw = ImageDraw.Draw(image)
            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))  # 用黑色填充
        return image

