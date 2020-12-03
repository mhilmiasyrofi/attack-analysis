#!/bin/bash
 
# Declare an array of string with type

declare -a train=("autoattack" "ffgsm" "mifgsm" "apgd" "pgd" "tpgd" )
declare -a test=("autoattack" "ffgsm" "mifgsm" "apgd" "pgd" "tpgd")


# Iterate the string array using for loop
for tr in ${train[@]}; do
    for ts in ${test[@]}; do
    python train_using_adv_example.py --model resnet20 \
        --train-adversarial $tr \
        --test-adversarial $ts \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 \
        --attack-iters 10 \
        --pgd-alpha 2 \
        --fname auto \
        --optimizer 'momentum' \
        --weight_decay 5e-4 \
        --batch-size 128 \
        --BNeval 
    done
done