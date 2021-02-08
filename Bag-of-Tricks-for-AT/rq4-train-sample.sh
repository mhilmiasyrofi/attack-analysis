#!/bin/bash

## Declare an array of string with type
# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# declare -a adv=("autoattack")

# # Iterate the string array using for loop
for a in ${adv[@]}; do
    python adversarial_training.py --model resnet18 \
        --attack $a \
        --chkpt-iters 1 \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --labelsmooth \
        --labelsmoothvalue 0.3 \
        --fname auto \
        --optimizer 'momentum' \
        --weight_decay 5e-4 \
        --batch-size 128 \
        --BNeval \
        --epochs 5 \
        --sample 50 \
        --val 1000
done