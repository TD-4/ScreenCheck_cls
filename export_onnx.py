# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

import argparse
import cv2
import os
import sys
import re
import time
import numpy as np
import datetime

import torch
from torchvision import transforms

import onnxruntime as ort

from models import model as models
from models import model_fixed as ScreenCheckModel


def export_onnx(args):
    # ----------------------------set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # ----------------------------load the image
    input_image = torch.zeros([1, 30, 224, 224], dtype=torch.float32).to(device)
    print("input size is..", input_image.shape)

    # ----------------------------load the model
    model = ScreenCheckModel.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model.train(False)
    model.load_state_dict((torch.load(args.model, map_location=device)))  # load our pre-trained model on the screencheck dataset
    print('Loading model_hyper network and weight...')
    pred = model(input_image)  # 0.8375

    # ----------------------------export the hyper network model
    input_names = ["input"]
    output_names = ["output"]
    print('exporting model to ONNX...')
    onnx_output_path = args.model[:-3]+"onnx"
    torch.onnx.export(model, input_image, onnx_output_path, verbose=True,
                      input_names=input_names, output_names=output_names, opset_version=10)
    print('model exported to {}'.format(onnx_output_path))


def compare_onnx_pth(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ------------------1、准备 images
    all_imgs = [img for img in os.listdir(args.image) if img[-4:] == ".bmp"]
    assert len(all_imgs) != 0, "{} is empty!".format(os.path.join(args.image))
    imgs = []
    # 读取图片
    for i, img_p in enumerate(all_imgs):
        img_path = [p for p in all_imgs if p.startswith(str(i)) and p.endswith(".bmp")]
        if len(img_path) == 0: continue
        imgs.append(cv2.imdecode(np.fromfile(os.path.join(args.image, img_path[0]), dtype=np.uint8), cv2.IMREAD_COLOR))
    while len(imgs) < 10:
        imgs.append(np.zeros((224, 224, 3), np.uint8))
    # 合并一组图片
    images = []
    for image in imgs:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(transforms.ToTensor()(image))
        images.append(image)
    input_img = torch.cat(images, 0)
    input_img = input_img.to(device=device).unsqueeze(0)

    # ---------------------------pth--------------------------------
    t_start = time.time()
    # 2、 准备 model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model_hyper.train(False)
    model_hyper.load_state_dict((torch.load(args.pth, map_location=device)))

    # 3、infer image of one group
    paras = model_hyper(input_img)  # 'paras' contains the network weights conveyed to target network
    # Building target network
    model_target = models.TargetNet(paras).cuda()
    for param in model_target.parameters():
        param.requires_grad = False
    # Quality prediction
    pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net

    pred_pth = pred.cpu().detach().numpy().argsort()[-2:][::-1]
    t_end = time.time()
    print("PTH模型的top1：{} , top2:{} 用时：{}".format(str(pred_pth[0]), str(pred_pth[1]), str(t_end-t_start)))

    # --------------------------onnx---------------------------------------
    t_start = time.time()
    # 2、准备模型
    sess = ort.InferenceSession(args.onnx)

    # 模型输入
    input_name = sess.get_inputs()[0].name
    # 模型输出
    output_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([output_name], {input_name: input_img.cpu().numpy()})[0]
    pred_pth = pred_onnx.argsort()[-2:][::-1]
    t_end = time.time()
    print("ONNX模型的top1：{} , top2:{} 用时：{}".format(str(pred_pth[0]), str(pred_pth[1]), str(t_end - t_start)))


if __name__ == "__main__":
    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/random_ScreenCheck_20211022150654_31.pth', help="set model checkpoint path")

    parser.add_argument('--image', type=str, default='test/20211018094229_3', help="image path")
    parser.add_argument('--pth', type=str, default='pretrained/random_ScreenCheck_20211022150654_31.pth', help="set model checkpoint path")
    parser.add_argument('--onnx', type=str, default='pretrained/random_ScreenCheck_20211022150654_31.onnx', help="set model checkpoint path")
    args = parser.parse_args()

    # 导出ONNX模型
    #export_onnx(args)

    # 对比pth与onnx的结果是否一致
    compare_onnx_pth(args)


