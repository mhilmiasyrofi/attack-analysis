#!/bin/bash

# Declare an array of string with type
declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# # Iterate the string array using for loop
for a in ${adv[@]}; do
    python adversarial_training.py --model resnet18 \
        --attack $a \
        --sample 75 \
        --chkpt-iters 1 \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 10 \
        --labelsmooth \
        --labelsmoothvalue 0.3 \
        --fname auto \
        --optimizer 'momentum' \
        --weight_decay 5e-4 \
        --batch-size 128 \
        --BNeval 
done