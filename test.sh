#!/usr/bin/env bash
echo "Start test......"
echo "First fold......"
python Train_GCN_Trans_AU.py --fold_train 1_2 --fold_test 3 --evaluate True --pretrained /home/ubuntu/userfile/data2/ssz/GT-AU-master/result/pretrained/BP4D/GCN_AU_train_1_2.pth
wait
echo "Second fold......"
python Train_GCN_Trans_AU.py --fold_train 1_3 --fold_test 2 --evaluate True --pretrained /home/ubuntu/userfile/data2/ssz/GT-AU-master/result/pretrained/BP4D/GCN_AU_train_1_3.pth
wait
echo "Third fold......"
python Train_GCN_Trans_AU.py --fold_train 2_3 --fold_test 1 --evaluate True --pretrained /home/ubuntu/userfile/data2/ssz/GT-AU-master/result/pretrained/BP4D/GCN_AU_train_2_3.pth