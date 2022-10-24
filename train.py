import os
from torch.nn.modules.loss import BCELoss
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from time import time
import sys
import logging
import argparse
from tqdm import tqdm

from source.dataset.data_list import ImagesList
from source.dataset.data_list import get_img
import source.dataset.pre_process as prep
from source.models.temporal_gcn_trans import Backbone_Model, Decoupling_Model, HSTR_AU, Fusion_Model
from source.config.config import get_config
from source.config.datasets import BP4D, DISFA
from source.functions import *

class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_dir()
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_dir(self):
        if self.args.dataset_id == 'BP4D':
            dataset = BP4D()
        else:
            dataset = DISFA()

        data_dir = dataset.dataset_path
        self.image_dir = dataset.image_path
        out_dir = dataset.out_dir
        self.num_AU_points = dataset.num_AU_points
        output_dir = os.path.join(out_dir, self.args.name + '_fold_' + self.args.fold_train)
        self.model_dir = os.path.join(output_dir, 'models')
        self.vis_dir = os.path.join(output_dir, 'vis')
        self.log_dir = os.path.join(output_dir, 'logs')
        self.res_dir = os.path.join(output_dir, 'res')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        # self.train_path = os.path.join(data_dir, self.args.dataset_id + '_crop_tr_' + self.args.fold_train + '.txt')
        self.train_path = os.path.join(data_dir, 'BP4D_temp.txt')
        # self.test_path = os.path.join(data_dir, self.args.dataset_id + '_crop_ts_' + self.args.fold_test + '.txt')
        self.test_path = os.path.join(data_dir, 'BP4D_temp.txt')
        self.corr_path = os.path.join(data_dir, self.args.dataset_id + '_corr_' + self.args.fold_train + '.txt')
        self.weight_path = os.path.join(data_dir, self.args.dataset_id + '_tr_' + self.args.fold_train + '_weight.txt')
        # self.land_path = os.path.join(data_dir, 'land.pkl')

        self.test_name = 'HSTR_fold_' + self.args.fold_test + '_seed_' + str(self.args.seed)
        self.fout_test_f1 = open(self.res_dir + '/' + self.test_name + '_f1.txt', 'w')
        self.fout_test_f1_mean = open(self.res_dir + '/' + self.test_name + '_f1_mean.txt', 'w')
        self.fout_test_acc = open(self.res_dir + '/' + self.test_name + '_acc.txt', 'w')
        self.fout_test_acc_mean = open(self.res_dir + '/' + self.test_name + '_acc_mean.txt', 'w')
        self.fout_test = open(self.res_dir + '/' + self.test_name + '_predata.txt', 'w')

        set_logging(self.log_dir)

    def init_environment(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def init_device(self):
        logging.info('use_gpu: ' + os.environ['CUDA_VISIBLE_DEVICES'])

    def init_dataloader(self):

        img_dic = get_img(self.image_dir)

        imDataset_train = ImagesList(crop_size=self.args.image_size, path=self.train_path, images_lenth=self.args.T_lenth,
                                    img_path=self.image_dir, img_dic=img_dic, NUM_CLASS=self.num_AU_points,
                                    transform=prep.image_train(crop_size=self.args.image_size),
                                    target_transform=prep.land_transform(img_size=self.args.image_size)
                                    )
        imDataset_test = ImagesList(crop_size=self.args.image_size, path=self.test_path, img_path=self.image_dir,
                                   img_dic=img_dic, NUM_CLASS=self.num_AU_points, phase='test', images_lenth=self.args.T_lenth,
                                   transform=prep.image_test(crop_size=self.args.image_size),
                                   target_transform=prep.land_transform(img_size=self.args.image_size)
                                   )



        self.imDataLoader_train = DataLoader(imDataset_train, batch_size=self.args.train_batch_size, shuffle=True,
                                  num_workers=self.args.num_workers)
        self.imDataLoader_test = DataLoader(imDataset_test, batch_size=self.args.test_batch_size, shuffle=False,
                                  num_workers=self.args.num_workers)

        logging.info('Dataset: {}'.format(self.args.dataset_id))
        logging.info('Batch size: train-{}, eval-{}'.format(self.args.train_batch_size,
                                                            self.args.test_batch_size))

    def init_model(self):
        self.model = HSTR_AU(self.args, self.num_AU_points, self.corr_path).cuda()
        logging.info('Model parameters: {:.2f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1000 / 1000))
        if self.args.evaluate:
            logging.info('Pretrained model: {}'.format(self.args.pretrained))
            self.model = torch.load(self.args.pretrained)[0]

    def init_optimizer(self):
        self.optimizer = SGD(self.model.get_config_optim(self.args.lr, self.args.lrp), lr=self.args.lr, momentum=self.args.momentum,
                        weight_decay=self.args.weight_decay)

    def init_loss_func(self):
        au_weight_src = torch.from_numpy(np.loadtxt(self.weight_path)).float().cuda()
        self.loss_func = nn.BCEWithLogitsLoss(au_weight_src).cuda()
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))

class Processor(Initializer):

    def CenterArcLoss(self, CenterLoss, x):
        N, C, d = x.shape
        index = torch.arange(C).unsqueeze(0).repeat(N,1).view(N*C)
        x = x.view(N*C, d)
        loss = CenterLoss(x, index)
        return loss

    def forward_process(self, model, datablob, y_lb):
        out_gcn, out_sac = model(datablob)
        loss = self.loss_func(out_gcn, y_lb) + \
               self.loss_func(out_sac, y_lb)

        outputs = out_gcn

        return loss, outputs

    def train(self, epoch):
        torch.enable_grad()
        self.model.train()
        start_train_time = time()

        iter = tqdm(self.imDataLoader_train, ncols=110)
        for batch_Idx, data in enumerate(iter):
            self.optimizer.zero_grad()
            datablob, datalb = data

            datablob = torch.autograd.Variable(datablob).cuda()

            y_lb = torch.autograd.Variable(datalb).view(datalb.size(0), -1).cuda()

            sum_loss, cls_pred = self.forward_process(self.model, datablob, y_lb)

            sum_loss.backward()

            cls_pred = cls_pred.data.cpu().float()
            y_lb = y_lb.data.cpu().float()
            f1_score = get_f1(cls_pred, y_lb)
            acc_scr = get_acc(cls_pred, y_lb)

            iter.set_description(
                'Loss: {:.3f}, Acc: {:.3f}, F1: {:.3f},'.format(sum_loss.cpu().data.item(),
                                                                acc_scr.mean().cpu().data.item(),
                                                                f1_score.mean().cpu().data.item()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_clip_grad_norm)
            self.optimizer.step()

            del datablob, y_lb, cls_pred, sum_loss, acc_scr, f1_score
            torch.cuda.empty_cache()

        new_model = self.model_dir + '/HSTR_AU_' + 'train_' + self.args.fold_train + '_epoch' + str(epoch+1) + '.pth'
        torch.save(self.model, new_model)
        logging.info('save ' + new_model)

        logging.info('Training time:'.format(get_time(time() - start_train_time)))
        logging.info('')

    @torch.no_grad()
    def test(self):
        torch.no_grad()
        self.model.eval()
        start_train_time = time()

        iter = tqdm(self.imDataLoader_test, ncols=110)
        for batch_Idx, data in enumerate(iter):
            datablob, datalb = data

            datablob = torch.autograd.Variable(datablob).cuda()

            y_lb = torch.autograd.Variable(datalb).view(datalb.size(0), -1).cuda()

            sum_loss, cls_pred = self.forward_process(self.model, datablob, y_lb)

            cls_pred = cls_pred.data.cpu().float()
            y_lb = y_lb.data.cpu().float()
            f1_score = get_f1(cls_pred, y_lb)
            acc_scr = get_acc(cls_pred, y_lb)

            if batch_Idx == 0:
                all_output = cls_pred
                all_label = y_lb
            else:
                all_output = torch.cat((all_output, cls_pred), 0)
                all_label = torch.cat((all_label, y_lb), 0)

            iter.set_description(
                'Loss: {:.3f}, Acc: {:.3f}, F1: {:.3f},'.format(sum_loss.cpu().data.item(),
                                                                acc_scr.mean().cpu().data.item(),
                                                                f1_score.mean().cpu().data.item()))
            self.fout_test.write('Label:' + str(y_lb) + '->' + 'Pre:' + str(cls_pred) + '\n')

            del datablob, y_lb, cls_pred, sum_loss, acc_scr, f1_score
            torch.cuda.empty_cache()

        all_acc_scr = get_acc(all_output, all_label)
        all_f1_score = get_f1(all_output, all_label)

        self.fout_test_f1.write('***' + str(all_f1_score.numpy().tolist()) + '\n')
        self.fout_test_f1_mean.write('***' + str(all_f1_score.mean().numpy().tolist()) + '\n')
        self.fout_test_acc.write('***' + str(all_acc_scr.numpy().tolist()) + '\n')
        self.fout_test_acc_mean.write('***' + str(all_acc_scr.mean().numpy().tolist()) + '\n')

        logging.info('Average f1 score: ' + str(all_f1_score.mean().numpy().tolist()))
        logging.info('Average acc score: ' + str(all_acc_scr.mean().numpy().tolist()))
        del all_acc_scr, all_f1_score, all_output, all_label

        logging.info('Test time: {}'.format(get_time(time() - start_train_time)))
        logging.info('')

    def start(self):
        start_time = time()
        if self.args.evaluate:
            # Loading Evaluating Model
            if not os.path.exists(self.args.pretrained):
                logging.error('No checkpoint load before evaluate !!!')
                raise ValueError()

            # Evaluating
            logging.info('Starting evaluating ...')
            self.test()
            logging.info('Finish evaluating!')

        else:
            logging.info('Starting training ...')
            for epoch in range(0, self.args.end_epoch):
                lr_change(epoch, self.args, self.optimizer)
                logging.info('Train for epoch {}/{} ...'.format(epoch + 1, self.args.end_epoch))
                self.train(epoch)
                logging.info('Test ...')
                self.test()

            logging.info('Total time: {}'.format(get_time(time() - start_time)))
            logging.info('Finish training!')
            logging.info('')

            self.fout_test_f1.close()
            self.fout_test_f1_mean.close()
            self.fout_test_acc.close()
            self.fout_test_acc_mean.close()
            self.fout_test.close()

def main():
    config = get_config()
    p = Processor(config)
    p.start()

if __name__ == '__main__':
    main()
    print('Done!')
