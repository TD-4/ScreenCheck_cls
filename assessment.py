# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

"""
模型评价工具
"""

from PyQt5 import QtWidgets,QtCore,QtGui
import cv2
import os
import sys
import re
import xlwt
import shutil
import time
import numpy as np
import datetime

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

    return pred_id


def write_excel_xls(path, sheet_name="", value=None):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")


class ImgTag(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 文件夹全局变量
        self.dataset_path = ""  # 数据集路径
        self.model_path = ""  # 数据集路径

        self.setWindowTitle("Assessment")
        # 主控件和主控件布局
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # 模型选择按钮
        self.select_model_btn = QtWidgets.QPushButton("选择模型")
        self.select_model_btn.clicked.connect(self.select_model_click)
        # 数据集选择按钮
        self.select_dataset_btn = QtWidgets.QPushButton("选择数据集")
        self.select_dataset_btn.clicked.connect(self.select_dataset_click)
        # 数据集选择按钮
        self.assess_btn = QtWidgets.QPushButton("Assessment")
        self.assess_btn.clicked.connect(self.assessment)

        # --------------状态栏--------------------------------
        self.status = QtWidgets.QLabel("   状态   ")
        # ----------------------------------------------------

        # 添加按钮到布局
        self.main_layout.addWidget(self.select_model_btn)
        self.main_layout.addWidget(self.select_dataset_btn)
        self.main_layout.addWidget(self.assess_btn)
        self.main_layout.addWidget(self.status)

        # 设置UI界面核心控件
        self.setCentralWidget(self.main_widget)

    def select_model_click(self):
        self.model_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择models')

    def select_dataset_click(self):
        self.dataset_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择dataset')

    def assessment(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        result_list = []    # 存放结果，最终写入到excel中

        models = [m for m in os.listdir(self.model_path) if m.endswith("pth")]
        num_models = len(models)
        datasets = os.listdir(self.dataset_path)

        # 准备表头 ： ID FolderName Label model...
        res_list = []
        res_list.append("ID")
        res_list.append("FolderName")
        res_list.append("Label")
        for i in range(0, num_models):  # models
            res_list.append(models[i])
        result_list.append(res_list)

        # 处理所有图片
        for i, img_group in enumerate(datasets):    # 每一组图片
            res_list = []
            res_list.append(i+1)    # 添加ID字段
            res_list.append(img_group)  # 添加FolderName字段
            res_list.append(img_group.split("_")[-1])  # 添加真实标签字段
            for model in models:  # 每一个模型
                pred = scores_one_group(os.path.join(self.dataset_path, img_group),  # 一组图片
                                        os.path.join(self.model_path, model),  # 一个模型
                                        device  # 设备
                                        )
                res_list.append(pred)   # 添加models预测值字段

            result_list.append(res_list)    # 将本组图片的处理结果存放到result_list中，供后续存入Excel
            # 设置状态栏 图片数量信息
            text_info = "img:{} - pred:{} .................... {} / {}".format(img_group, str(res_list[-1]), i+1, str(len(datasets)))
            self.status.setText(text_info)
            self.status.repaint()
            QtWidgets.QApplication.processEvents()
            print("处理完 {} ---- {}".format(img_group, str(res_list[-1])))
        # 写入excel文件
        dest_path = os.path.join("pretrained", datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".xls")
        write_excel_xls(dest_path, sheet_name="assess", value=result_list)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ImgTag()
    gui.resize(800,400)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
