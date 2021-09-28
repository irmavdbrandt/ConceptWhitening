#!/bin/sh
python train_places.py --workers 0 --arch resnet_cw --depth 18 --epochs 200 --batch-size 12 --lr 0.05 --whitened_layers 7 --concepts test2 --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER val_256/
