#!/usr/bin/env bash
echo "Start run......"
echo "First fold......"
python train.py --fold_train 1_2 --fold_test 3 --name $1
echo "Second fold......"
python train.py --fold_train 1_3 --fold_test 2 --name $1
echo "Third fold......"
python train.py --fold_train 2_3 --fold_test 1 --name $1
echo "Run complete!"