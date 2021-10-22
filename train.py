# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import datetime
import argparse
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch

from dataloader.screenCheck import ScreenCheckDataLoader, DataPrefetcher
from models.model import HyperNet, TargetNet


class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.device_count() > 0 else 'cpu')
        # 1、数据加载
        kwargs = {
            'data_dir': args.dataset,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'augment': False,  # 是否进行数据增强
            'in_channels': 30,  # 图片的通道数
            'val': False,  # 是否是验证集

            "batch_size": args.batchsize,
            "num_workers": 4,
            "shuffle": True,
            "drop_last": True
        }
        self.train_loader = DataPrefetcher(ScreenCheckDataLoader(**kwargs), device=self.device)
        kwargs['val'] = True
        #kwargs['batch_size'] = 4
        self.val_loader = DataPrefetcher(ScreenCheckDataLoader(**kwargs), device=self.device)

        # 2、模型
        self.model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(self.device)  # .cuda()  # Hyper Network, 包含backbone
        self.model_hyper.train(True)

        # 5、迁移学习
        if args.resume is not None:
            save_model = torch.load(args.resume)
            print("load weight from {}".format(args.resume))
            model_dict = self.model_hyper.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model_hyper.load_state_dict(model_dict)

        # 3、loss
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.CE = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')

        # 4、优化器
        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = args.lr
        self.lrratio = args.lr_ratio
        self.weight_decay = args.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.optim = torch.optim.Adam(paras, weight_decay=self.weight_decay)

    # 6、训练
    def train(self):
        """Training all epochs"""
        best_srcc = 0.0
        best_plcc = 0.0
        # epochs
        for t in range(self.epochs):  # epochs
            epoch_loss = []
            #pred_scores = []
            #gt_scores = []
            top1 = []
            top2 = []
            print("\n............................epoch  {} ..............".format(str(t+1)))
            tbar = tqdm(self.train_loader, 'train', ncols=130)
            for img, label, img_p in tbar:  # iters
                self.optim.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                #pred_scores = pred_scores + pred.cpu().tolist()
                #gt_scores = gt_scores + label.cpu().tolist()
                top1 = top1 + [self.eval_metrics(pred,label,topk=(1,2))[0].item()]
                top2 = top2 + [self.eval_metrics(pred,label,topk=(1,2))[1].item()]

                #loss = self.l1_loss(pred.squeeze(), label.float().detach())
                loss = self.CE(pred, label)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optim.step()
            top1_mean, top2_mean = np.mean(top1), np.mean(top2)

            #test_srcc, test_plcc = self.test()
            test_top1, test_top2 = self.test()
            if test_top1 > best_srcc:  # 当前epoch的效果是否大于best，是则保存pth；否则，无操作
                best_srcc = test_top1
                #best_plcc = test_plcc
                print("Saving pth .........")
                torch.save(self.model_hyper.state_dict(), os.path.join(self.args.store, "ScreenCheck_{}_{}.pth".format(str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), t)))
            print('Epoch\tTrain_Loss\tTrain_Top1\tTest_top1\tTest_top2')
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %  (t + 1, sum(epoch_loss) / len(epoch_loss), top1_mean, test_top1, test_top2))

            # Update optimizer
            lr = self.lr / pow(10, (t // 20))   # 每20epochs, 降低lr = lr/10
            if t > 8:
                self.lrratio = 1
            paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                     {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                     ]
            self.optim = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self):
        """Testing"""
        self.model_hyper.train(False)
        # pred_scores = []
        # gt_scores = []
        top1 = []
        top2 = []

        for img, label, img_p in tqdm(self.val_loader, 'val', ncols=130):
            paras = self.model_hyper(img)
            model_target = TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            #pred_scores = pred_scores + [pred.cpu().detach().numpy().item()]  # pred_scores.append(float(pred.item()))
            #gt_scores = gt_scores + label.cpu().tolist()
            top1 = top1 + [self.eval_metrics(pred, label, topk=(1, 2))[0].item()]
            top2 = top2 + [self.eval_metrics(pred, label, topk=(1, 2))[1].item()]

        # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        # test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        # test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        top1_mean, top2_mean = np.mean(top1), np.mean(top2)

        self.model_hyper.train(True)
        # return test_srcc, test_plcc
        return top1_mean, top2_mean

    def eval_metrics(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ScreenCheck PyTorch Training')
    parser.add_argument('--dataset', default=r'/root/data/datasets/iqa/9_ScreenCheck_20211021_merge', type=str, help='dataset path')

    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate。主干网络的学习率')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network。主干网络*lr_ratio是hypernet的学习率')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=96, help='Batch size。 batchsize大小')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='Epochs for training')
    parser.add_argument('--resume', dest='resume', type=str, default=None, help='weight from other dataset')
    parser.add_argument('--store', dest='store', type=str, default="pretrained/20211022_merge_cls", help=' save path')
    parser.add_argument('--cuda', dest='cuda', type=str, default="0", help=' save path')
    args = parser.parse_args()

    solver = HyperIQASolver(args)
    solver.train()

