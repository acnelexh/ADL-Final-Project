#!/bin/bash

# This script is used to train the model

python train.py \
   --dataset ./data/exported-traced-adjacencies \
   --model SimpleGNN \
   --num_classes 6224 \
   --output_dir ./runs \
   --tensorboard_log_dir ./tensorboard_log \
   --lr 0.01 \
   --lr_scheduler multisteplr \
   --lr_steps 10 20 30 40 \
   --lr_gamma 0.5 \
   --lr_decay 1e-05 \
   --epochs 50 \
   --batch_size 32 \
   --weight_decay 0.0001 \
   --warm_up None \
   --num_workers 0 \
   --device cuda \
   --print_freq 1 \
   --save_model_interval 1000 \
   --log_epochs 1 \
   --hidden_dim 32 128 1024

