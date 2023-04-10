#!/bin/bash

# This script is used to train the model

python train.py\
    --dataset ./data/exported-traced-adjacencies/\
    --device cpu\
    --num_classes 6224