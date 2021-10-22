# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

"""
图片评分工具
"""

from PyQt5 import QtWidgets,QtCore,QtGui
import sys,os


class ImgTag(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 文件夹全局变量
        self.dir_path = ""  # 文件夹路径，此路径下包含多组图片，每组图片是个文件夹
        self.img_index_dict = dict()    # 所有组 的id、名称字典
        self.img_index_dict2 = dict()   # 所有组 的名称、修改后名称字典
        self.current_index = 0  # 当前组的ID
        self.current_filename = ""  # 当前组的名称
        self.current_filename_img_num = -1  # 当前组的图片数量

        self.setWindowTitle("IQA图片标注")
        # 主控件和主控件布局
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # -------------------图像展示控件----------------
        self.imgs_widget = QtWidgets.QWidget()
        self.imgs_layout = QtWidgets.QGridLayout()
        self.imgs_widget.setLayout(self.imgs_layout)

        self.imgs_view = {}
        self.imgs_name = {}
        self.imgs_input = {}
        loc = [(0,0), (0,1), (0,2), (0,3),
               (3,0), (3,1), (3,2), (3,3),
               (6,0), (6,1), (6,2), (6,3)]
        for i in range(0, 12):
            self.imgs_view[str(i)] = QtWidgets.QLabel("   图片{}占位符    ".format(str(i)))  # 标签占位, 或图片view
            self.imgs_view[str(i)].setAlignment(QtCore.Qt.AlignCenter)
            self.imgs_name[str(i)] = QtWidgets.QLabel()  # 图像名称
            # self.imgs_input[str(i)] = QtWidgets.QLineEdit()  # 图像标注控件，或文本框

            self.imgs_layout.addWidget(self.imgs_view[str(i)], loc[i][0], loc[i][1])
            self.imgs_layout.addWidget(self.imgs_name[str(i)], loc[i][0] + 1, loc[i][1])
            # self.imgs_layout.addWidget(self.imgs_input[str(i)], loc[i][0] + 2, loc[i][1])
        # --------------------------------------------------------

        # ---------------控制按钮控件-------------------------------
        self.opera_widget = QtWidgets.QWidget()
        self.opera_layout = QtWidgets.QVBoxLayout()
        self.opera_widget.setLayout(self.opera_layout)
        # 各个按钮
        self.select_name = QtWidgets.QLabel("第 ？ 张图片是缺陷")
        self.select_input = QtWidgets.QLineEdit()  # 图像标注控件，或文本框

        self.select_img_btn = QtWidgets.QPushButton("选择目录")
        self.select_img_btn.clicked.connect(self.select_img_click)

        self.previous_img_btn = QtWidgets.QPushButton("上一张")
        self.previous_img_btn.setEnabled(False)
        self.previous_img_btn.setShortcut('Ctrl+f')
        self.previous_img_btn.clicked.connect(self.previous_img_click)

        self.next_img_btn = QtWidgets.QPushButton("下一张")
        self.next_img_btn.setEnabled(False)
        self.next_img_btn.setShortcut('Ctrl+d')
        self.next_img_btn.clicked.connect(self.next_img_click)

        self.save_img_btn = QtWidgets.QPushButton("保存")
        self.save_img_btn.setEnabled(False)
        self.save_img_btn.setShortcut('Ctrl+s')
        self.save_img_btn.clicked.connect(self.next_img_click)

        # 添加按钮到布局
        self.opera_layout.addWidget(self.select_name)
        self.opera_layout.addWidget(self.select_input)
        self.opera_layout.addWidget(self.select_img_btn)
        self.opera_layout.addWidget(self.previous_img_btn)
        self.opera_layout.addWidget(self.next_img_btn)
        self.opera_layout.addWidget(self.save_img_btn)
        # ----------------------------------------------------

        # ------------将控件添加到主控件布局层--------------------
        self.main_layout.addWidget(self.imgs_widget, 0, 0)
        self.main_layout.addWidget(self.opera_widget, 0, 12)
        # ---------------------------------------------------

        # --------------状态栏--------------------------------
        self.img_total_current_label = QtWidgets.QLabel()
        self.img_total_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.img_total_current_label)
        self.statusBar().addPermanentWidget(self.img_total_label, stretch=0)  # 在状态栏添加永久控件
        # ----------------------------------------------------

        # 设置UI界面核心控件
        self.setCentralWidget(self.main_widget)

    def save_imgs(self):
        self.img_index_dict2[self.img_index_dict[self.current_index]] = self.img_index_dict[self.current_index] + "----" + str(self.select_input.text())
        return True

    def refresh(self):
        # 刷新图片显示部分——实例化12个图像
        all_images = os.listdir(os.path.join(self.dir_path, self.current_filename))
        for img_p in all_images:
            image = QtGui.QPixmap(os.path.join(self.dir_path, self.current_filename, img_p)).scaled(250, 250)
            if img_p[0].isnumeric():
                i = img_p[0]
            else:
                i = img_p[-5]
            # 1、显示图像
            self.imgs_view[i].setPixmap(image)
            # 2、显示图片名称、并保存名称
            self.imgs_name[i].setText(img_p)  # 显示文件名

        # 3、显示文本框，获取分数
        folder_name = self.img_index_dict[self.current_index]
        if self.img_index_dict2[folder_name] !="":
            folder_name = self.img_index_dict2[folder_name]
        split_imgp = folder_name.split('----')
        if len(split_imgp) >= 2:  # 已经标注过的，获取分数
            score = split_imgp[-1]
        else:  # 没有标注过得显示空
            score = ""
        self.select_input.setText(score)
        self.select_input.setFocus()  # 获取输入框焦点
        self.select_input.selectAll()  # 全选文本

        # 设置状态栏 图片数量信息
        self.img_total_current_label.setText("{}".format(self.current_index + 1))
        self.img_total_label.setText("/{total}".format(total=len(os.listdir(self.dir_path))))

    def checkout_dir(self):
        # 修改文件夹中，1-Z脉冲1630.bmp，命名不正确的情况
        all_images = os.listdir(os.path.join(self.dir_path, self.current_filename))
        tmp = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")
        for img_p in all_images:
            if (img_p[0] in set(tmp) or img_p[-5] in set(tmp)) and img_p[-4:] == ".bmp":
                pass
            else:
                # str_ = self.current_filename + "文件夹中，有文件命名不正确, 即将删除此文件 " + img_p
                # QtWidgets.QMessageBox.information(self, 'Warning', str_, QtWidgets.QMessageBox.Yes,
                #                                   QtWidgets.QMessageBox.Cancel)
                os.remove(os.path.join(self.dir_path, self.current_filename, img_p))

    def _reset(self):
        try:
            for i in range(0, 12):
                self.imgs_view[str(i)].setText("   图片{}占位符    ".format(str(i)))
                self.imgs_name[str(i)].setText("")
        except Exception as e:
            print(e)

    # 选择目录按钮
    def select_img_click(self):
        self._reset()
        try:
            self.dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹')
            dir_list = os.listdir(self.dir_path)
            if len(dir_list) <= 0:
                QtWidgets.QMessageBox.information(self, '提示', '文件夹没有发现图片文件！', QtWidgets.QMessageBox.Ok)
                return

            # 建立“选择目录“下所有文件夹索引
            for i, d in enumerate(dir_list):
                self.img_index_dict[i] = d
                self.img_index_dict2[d] = ""
            # 当前的文件夹索引
            self.current_index = 0
            # 当前文件夹路径
            self.current_filename = self.img_index_dict[self.current_index]
            self.setWindowTitle(self.img_index_dict[self.current_index])    # 修改窗口的名称为文件夹

            # 检查当前文件夹中文件是否符合要求
            self.checkout_dir()
            self.current_filename_img_num = len(os.listdir(os.path.join(self.dir_path, self.current_filename)))
            # 刷新图片显示部分 & 状态栏
            self.refresh()

            # 启用其他按钮
            self.previous_img_btn.setEnabled(True)
            self.next_img_btn.setEnabled(True)
            self.save_img_btn.setEnabled(True)
        except Exception as e:
            print(e)

    # 下一个文件夹
    def next_img_click(self):
        # 保存标注内容
        if self.save_imgs():
            # 判断是否越界
            if self.current_index == len(os.listdir(self.dir_path))-1:
                QtWidgets.QMessageBox.information(self, 'Warning', "已经是最后一张了", QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                return
            # 清空界面缓存
            self._reset()
            # 当前图像索引加1
            self.current_index += 1
            if self.current_index in self.img_index_dict.keys():
                # 当前图片文件路径
                self.current_filename = self.img_index_dict[self.current_index]
                self.setWindowTitle(self.current_filename)
                # 检查当前文件夹中文件是否符合要求
                self.checkout_dir()
                self.current_filename_img_num = len(os.listdir(os.path.join(self.dir_path, self.current_filename)))
                # 刷新页面
                self.refresh()

    # 上一个文件夹
    def previous_img_click(self):
        # 重命名，保存评分
        if self.save_imgs():
            # 判断是否越界
            if self.current_index == 0:
                QtWidgets.QMessageBox.information(self, 'Warning', "已经是第一张了", QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                return
            # 清空界面缓存
            self._reset()
            # 当前图像索引减1
            self.current_index -= 1
            if self.current_index in self.img_index_dict.keys():
                # 当前图片文件路径
                self.current_filename = self.img_index_dict[self.current_index]
                self.setWindowTitle(self.current_filename)
                # 检查当前文件夹中文件是否符合要求
                self.checkout_dir()
                self.current_filename_img_num = len(os.listdir(os.path.join(self.dir_path, self.current_filename)))
                # 刷新页面
                self.refresh()

    def closeEvent(self, event):
        self.save_imgs()
        for tmp in self.img_index_dict.keys():
            if self.img_index_dict2[self.img_index_dict[tmp]] != "":
                os.rename(os.path.join(self.dir_path, self.img_index_dict[tmp]),
                          os.path.join(self.dir_path, self.img_index_dict2[self.img_index_dict[tmp]]))

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ImgTag()
    gui.show()
    sys.exit(app.exec_())




if __name__ == '__main__':
    main()
