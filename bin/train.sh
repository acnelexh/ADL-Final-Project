#!/bin/bash

# This script is used to train the model

python train.py\
    --dataset ./data/exported-traced-adjacencies/\
    --device cuda\
    --num_classes 6224 \
    --lr 1e-3\
    --save_model_interval 25\
    --num_workers 4\
    --epochs 50\
    --batch_size 512