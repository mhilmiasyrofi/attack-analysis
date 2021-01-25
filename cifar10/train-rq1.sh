#!/bin/bash
 
## docker run -it --rm --name gpu0-bot -v /home/mhilmiasyrofi/Documents/Bag-of-Tricks-for-AT/:/workspace/Bag-of-Tricks-for-AT/ --gpus '"device=0"' mhilmiasyrofi/advtraining
 
## Declare an array of string with type
declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

## Used to run in different GPUs
# declare -a adv=("autoattack" "autopgd" "bim")
# declare -a adv=("cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack")
# declare -a adv=("spatialtransformation" "squareattack")

## Iterate the string array using for loop
for a in ${adv[@]}; do
    python adversarial_training.py --model resnet18 \
        --attack $a \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 \
        --labelsmooth \
        --labelsmoothvalue 0.3 \
        --fname auto \
        --optimizer 'momentum' \
        --weight_decay 5e-4 \
        --batch-size 128 \
        --BNeval 
done
