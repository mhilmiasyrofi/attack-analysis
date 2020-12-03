#!/bin/bash
 
# Declare an array of string with type
# declare -a adv=("apgd" "autoattack" "deepfool" "ffgsm" "fgm" "mifgsm" "pgd" "spatialtransformation" "squareattack" "tpgd")

declare -a adv=("apgd" "autoattack" "deepfool" "ffgsm" "fgm")

# declare -a adv=("mifgsm" "pgd" "spatialtransformation" "squareattack" "tpgd")


# Iterate the string array using for loop
for a in ${adv[@]}; do
    python adversarial_training.py --model resnet18 \
        --adversarial-data $a \
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

# python adversarial_training.py --model resnet18 \
#     --adversarial-data ffgsm \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 3 \
#     --attack-iters 10 \
#     --pgd-alpha 2 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval 