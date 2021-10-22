# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520


import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class ScreenCheckDataLoader(DataLoader):
    def __init__(self,
                 data_dir=None, mean=None, std=None, augment=False, in_channels=30, val=False,  # dataset
                 batch_size=1, num_workers=1, shuffle=True, drop_last=True  # dataloader
                 ):
        kwargs = {
            'root': data_dir,
            'mean': mean,
            'std': std,
            'augment': augment,      # 是否进行数据增强
            'in_channels': in_channels,     # 图片的通道数
            'val': val      # 是否是验证集
        }

        self.dataset = ScreenCheckDataset(**kwargs)

        super(ScreenCheckDataLoader, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


class ScreenCheckDataset(Dataset):
    def __init__(self, root=None, mean=None, std=None, augment=False, in_channels=30, val=False):
        # 图像路径
        self.root = root
        # 是否是验证集
        self.val = val

        # Normalization
        self.mean = mean
        self.std = std

        # 数据增强
        self.augment = augment

        # 输入图像的通道数，是否是灰度图像
        self.in_channels = in_channels

        # 所有文件的路径和标签
        self.files = []
        self._set_files()  # 获取所有文件的路径和标签，存放到files中

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        images, label, image_path = self._load_data(index)  # images:ndarray(bgr)->list of one group, label:int, string
        imgs = []
        for image in images:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            if self.val:  # 验证集
                image = self._val_augmentation(image)   # augmentation
            elif self.augment:  # 训练集
                image = self._augmentation(image)  # augmentation
            image = self.normalize(self.to_tensor(image))
            imgs.append(image)
        imgs = torch.cat(imgs, 0)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return imgs, label, image_path

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

    def _set_files(self):
        """
        功能：获取所有文件的文件名和标签
        """
        if self.val:
            list_path = os.path.join(self.root, "vallist.txt")
        else:
            list_path = os.path.join(self.root, "trainlist.txt")

        images, labels = [], []
        with open(list_path, 'r') as images_labels:
            for image_label in images_labels:
                images.append(image_label.strip())
                labels.append(image_label.strip().split("_")[-1])

        self.files = list(zip(images, labels))

    def _load_data(self, index):
        """
        功能：通过文件名获得，图片和类别
        :param index:
        :return:
        """
        image_path, label = self.files[index]
        assert self.in_channels >= 30, "输入channels不正确"

        img = []
        for i, img_p in enumerate(os.listdir(os.path.join(self.root, image_path))):
            img.append(cv2.imdecode(np.fromfile(os.path.join(self.root, image_path, str(i)+".bmp"), dtype=np.uint8), cv2.IMREAD_COLOR))
        while len(img) < self.in_channels // 3:
            img.append(np.zeros((224, 224, 3), np.uint8))
        return img, label, image_path

    def _val_augmentation(self, image):
        return image

    def _augmentation(self, image):
        # # Rotate the image with an angle between -10 and 10
        # h, w, _ = image.shape
        # angle = random.randint(-10, 10)
        # center = (w / 2, h / 2)
        # rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # image = cv2.warpAffine(image, rot_matrix, (w, h),
        #                        flags=cv2.INTER_LINEAR)  # , borderMode=cv2.BORDER_REFLECT)
        #
        # # Random H flip
        # if random.random() > 0.5:
        #     image = np.fliplr(image).copy()
        #
        # # Gaussian Blud (sigma between 0 and 1.5)
        # sigma = random.random()
        # ksize = int(3.3 * sigma)
        # ksize = ksize + 1 if ksize % 2 == 0 else ksize
        # image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
        #                          borderType=cv2.BORDER_REFLECT_101)

        return image


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        """
        self.train_loader = DataPrefetcher(train_loader, device=self.device)
        """
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.next_image_path = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_image_path = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_image_path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            self.next_image_path = self.next_image_path

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            image_path = self.next_image_path
            self.preload()
            count += 1
            yield input, target, image_path
            if type(self.stop_after) is int and (count > self.stop_after):
                break


if __name__ == "__main__":
    kwargs = {
        'data_dir': r"E:\screencheck",
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'augment': True,  # 是否进行数据增强
        'in_channels': 30,  # 图片的通道数
        'val': False,  # 是否是验证集

        "batch_size":4,
        "num_workers":2,
        "shuffle":True
    }
    train_loader = ScreenCheckDataLoader(**kwargs)
    for data, target, image_path in train_loader:
        print(data)  # torch.Size([4, 30, 224, 224])
        print(target)   # tensor([2, 1, 1, 1])
        print(image_path)   # tuple('20211013164736_2', '20211013164602_1', '20211013164708_1', '20211013164521_1')
