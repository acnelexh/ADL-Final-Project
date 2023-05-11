#!/bin/bash

python train.py \
   --dataset ./data/exported-traced-adjacencies \
   --model SimpleGNN \
   --output_dir ./runs \
   --tensorboard_log_dir ./tensorboard_log \
   --lr 0.001 \
   --lr_scheduler multisteplr \
   --lr_step_size 20 \
   --lr_steps 16 22 40 \
   --lr_gamma 0.1 \
   --lr_decay 1e-05 \
   --epochs 50 \
   --batch_size 32 \
   --weight_decay 0.0001 \
   --warm_up None \
   --seed 1 \
   --hidden_dim 512 512 \
   --input_dim 6 \
   --label_embed_dim 256 \
   --random_split False \
   --label_weight 1.0 \
   --unlabel_weight 1.0 \
   --normalize True \
   --proportion 0.1 \
   --sample_method random \
   --num_heads 4 \
   --edge_weight True \
   --topological_feature False \
   --few_shot True \
   --class_balance_loss False \
   --num_workers 0 \
   --data_split 0.6 0.2 0.2 \
   --device cuda \
   --print_freq 20 \
   --save_model_interval 10 \
   --log_epochs 1 \
