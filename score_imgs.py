# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520


import os
import cv2
import shutil
import time
import numpy as np

import torch
from torchvision import transforms
from models import model as models


def scores_one_group(path="", model_pth="", device=None):
    # ------------------1、准备 images
    all_imgs = [img for img in os.listdir(path) if img[-4:]==".bmp"]
    assert len(all_imgs) != 0, "{} is empty!".format(os.path.join(path))
    imgs = []
    # 读取图片
    for i, img_p in enumerate(all_imgs):
        img_path = [p for p in all_imgs if p.startswith(str(i)) and p.endswith(".bmp")]
        if len(img_path) == 0:continue
        imgs.append(cv2.imdecode(np.fromfile(os.path.join(path, img_path[0]), dtype=np.uint8), cv2.IMREAD_COLOR))
    while len(imgs) < 10:
        imgs.append(np.zeros((224, 224, 3), np.uint8))
    # 合并一组图片
    images = []
    for image in imgs:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(transforms.ToTensor()(image))
        images.append(image)
    input_img = torch.cat(images, 0)

    # ------------------2、 准备 model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model_hyper.train(False)
    model_hyper.load_state_dict((torch.load(model_pth, map_location=device)))

    # ------------------3、infer image of one group
    start_time = time.time()
    input_img = input_img.to(device=device).unsqueeze(0)
    paras = model_hyper(input_img)  # 'paras' contains the network weights conveyed to target network

    # Building target network
    model_target = models.TargetNet(paras).cuda()
    for param in model_target.parameters():
        param.requires_grad = False

    # Quality prediction
    pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
    pred = float(pred.item())
    pred_id = round(pred)

    # ------------------4、重命名文件夹
    os.rename(os.path.join(path), os.path.join(path)+"----" + str(pred_id) )#+ "-"+ str(pred)


if __name__ == "__main__":
    root_ = r"D:\2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for folder_p in os.listdir(root_):
        print("处理 {}".format(folder_p))
        scores_one_group(
            path=os.path.join(root_, folder_p),
            model_pth=r"pretrained/ScreenCheck_20211014184110_172.pth",
            device=device)
