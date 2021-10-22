import os
import re
import time
from loguru import logger
import shutil
import random
import datetime
import numpy as np
import cv2


def change_folder_name_by_ref(path=""):
    for folder in os.listdir(os.path.join(path)):
        ref_img = None
        for img_p in os.listdir(os.path.join(path,folder)):
            if re.search("_ref", img_p):
                ref_img = img_p.split("_")[0]
        os.rename(os.path.join(path, folder),
                  os.path.join(path, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "_" + str(ref_img)))
        # 休眠1秒钟
        time.sleep(1)


def change_folder_name_by_line(path=""):
    for folder in os.listdir(os.path.join(path)):
        if len(folder.split("----")) == 2 and folder.split("----")[1].isnumeric():
            ref_img = folder.split("----")[1]
            os.rename(os.path.join(path, folder),
                      os.path.join(path, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "_" + str(ref_img)))
        else:
            #os.remove(os.path.join(path, folder))
            pass
        # 休眠1秒钟
        time.sleep(1)


def change_one_image(path=""):
    for folder in os.listdir(os.path.join(path)):
        if len(folder.split("_")) !=2:continue
        for img_p in os.listdir(os.path.join(path, folder)):
            if img_p[0].isnumeric():
                i = img_p[0]
            else:
                i = img_p[-5]
            os.rename(os.path.join(path,folder,img_p),
                      os.path.join(path,folder,str(i) + ".bmp"))


def gen_train_val_list(path=""):
    all_groups = [group for group in os.listdir(path) if os.path.isdir(os.path.join(path, group))]
    sel_num = list(range(0, len(all_groups)))
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.9 * len(sel_num)))]  # train数据集id数量
    val_index = sel_num[int(round(0.9 * len(sel_num))):len(sel_num)]  # test数据集id数量

    with open(os.path.join(path, "trainlist.txt"), "a+") as train_file:
        for train_i in train_index:
            folder_name = all_groups[train_i]
            train_file.write(folder_name + "\n")
    with open(os.path.join(path, "vallist.txt"), "a+") as val_file:
        for val_i in val_index:
            folder_name = all_groups[val_i]
            val_file.write(folder_name + "\n")


def check_dataset(path=""):
    for folder in os.listdir(os.path.join(path)):
        print(".", end="")
        temp = [im[0] for im in os.listdir(os.path.join(path, folder)) if im[-4:] == ".bmp"]
        for i, img_p in enumerate(os.listdir(os.path.join(path, folder))):
            if str(i) not in temp:
                print(os.path.join(folder))

def aug_data(path=""):
    # 先扩充每一组为10张图片
    for folder in os.listdir(os.path.join(path)):
        temp = [im[:-4] for im in os.listdir(os.path.join(path, folder)) if im[-4:] == ".bmp"]
        img_h = 0
        img_w = 0
        # 检查此时文件夹中的数据是否是0到9，按顺序的
        for i, img_p in enumerate(os.listdir(os.path.join(path, folder))):
            image = cv2.imdecode(np.fromfile(os.path.join(path,folder, img_p),dtype=np.uint8),-1)
            img_h  = image.shape[0]
            img_w  = image.shape[1]
            if str(i) not in temp:
                print(os.path.join(folder), " 文件排序错误！！！")

        add_id = len(temp)
        if add_id < 10:
            img = np.zeros((img_h, img_w, 3), np.uint8)
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(path, folder, str(add_id) + ".bmp"))
            add_id +=1

    total_num = 0
    all_group = os.listdir(path)
    for one_group in all_group:  # 一组
        real_label = one_group.split("_")[-1]   # eg. 1
        all_images_p = os.listdir(os.path.join(path, one_group))
        add_num = 0
        for _ in range(100):     # 增强10次
            random.shuffle(all_images_p)
            random.shuffle(all_images_p)
            aug_label = "tmp"
            aug_path = os.path.join(path, one_group.split("_")[0] + "_" + aug_label)
            os.mkdir(aug_path)
            for i, img_p in enumerate(all_images_p):
                shutil.copy(os.path.join(path, one_group, img_p), os.path.join(aug_path, str(i)+".bmp"))
                if real_label == img_p[:-4]:
                    aug_label = str(i)

            # 是否保留增强内容
            if aug_label == real_label:
                # 删除
                shutil.rmtree(os.path.join(aug_path))
            elif aug_label in [m.split("_")[-1] for m in os.listdir(os.path.join(path)) if m.startswith(one_group.split("_")[0])]:
                shutil.rmtree(os.path.join(aug_path))
            else:
                os.rename(aug_path,aug_path.split("_")[0]+"_"+ aug_label)
                add_num += 1
        print("{} 文件夹增加了 {}".format(one_group, add_num))
        total_num += add_num
    print("----- 文件夹增加了 {}-----".format(total_num))


def check_trainval_txt(path=""):
    for txt_p in [t for t in os.listdir(path) if t.endswith("txt")]:
        with open(os.path.join(path, txt_p), "r") as train_or_val_txtfile:
            for image_label in train_or_val_txtfile:
                assert int(image_label.split("_")[1])<10, print("{} Error---".format(image_label))



if __name__ == "__main__":
    path = r"D:\Downloads\点灯复核数据集\10_ScreenCheck_20211021_merge_random"
    # -------------修改文件夹名--------------------
    # 1、原来数据集是按照每张图评分的，而且最高分后缀为“_ref"，所以此函数是将文件夹修改名称，把分数最高的ID放到文件夹名字的后面
    # change_folder_name_by_ref(r"D:\trainval - 副本 (2)\20210922_LaiBao")
    # 1、2021-09-26 22.03.32.482_治具1_屏0__白画面缺陷_BlackDot缺陷_X1664_Y994_23040676TXP179100SY6179XXXXN179E179XXXXX----8
    #change_folder_name_by_line(path)

    # ------------修改图片名----------------------
    # 2、修改文件夹中每张图片，使之变成id.bmp
    #change_one_image(path)

    #check_dataset(path)

    # 3、增强
    # aug_data(path)
    # 3、生成train val list
    #gen_train_val_list(path)

    check_trainval_txt(path)
    pass
