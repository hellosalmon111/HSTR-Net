import os
import argparse
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(description='Pytorch_HSTR')

    # Model Configuration
    parser.add_argument('--model_name', type=str, default='HSTR')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--image_size', type=int, default=224, help='origin size for images')
    parser.add_argument('--dataset_id', type=str, default='BP4D')
    parser.add_argument('--fold_train', type=str, default='1_2')
    parser.add_argument('--fold_test', type=str, default='3')
    parser.add_argument('--name', type=str, default='ATT_GCN-trans_sac+out_selfattout_b16')
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--train_batch_size', default='16', type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default='10', type=int, help='test batch size')
    parser.add_argument('--T_lenth', default='5', type=int, help='test batch size')
    parser.add_argument('--end_epoch', type=int, default=10)
    parser.add_argument('--seed', default='1', type=int, help='random seed number') 
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)
    parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP',
                        help='learning rate for pre-trained layers')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--dim', type=int, default=1024, help='dim')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--max_clip_grad_norm', default=1.0, type=float, metavar='M', help='max_clip_grad_norm')
    parser.add_argument('--epoch_step', default=[5, 10], type=int, nargs='+',
                        help='number of epochs to change learning rate')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pretrained', type=str, default='./result/pretrained/BP4D/fold1.pth')
    parser.add_argument('--t', default=0.4, type=float)
    parser.add_argument('--num_heads', type=int, default=4)

    cfg = parser.parse_args()
    gpu = cfg.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    return cfg
