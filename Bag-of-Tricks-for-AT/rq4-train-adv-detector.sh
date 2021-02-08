#!/bin/bash
 
## docker run -it --rm --name gpu0-bot -v /home/mhilmiasyrofi/Documents/Bag-of-Tricks-for-AT/:/workspace/Bag-of-Tricks-for-AT/ --gpus '"device=0"' mhilmiasyrofi/advtraining
 
## Declare an array of string with type
# declare -a adv=("pixelattack_spatialtransformation_bim" "pixelattack_spatialtransformation_deepfool" "pixelattack_spatialtransformation_autoattack")

# declare -a adv=("pixelattack_spatialtransformation_bim")

# declare -a adv=("pixelattack_spatialtransformation_autoattack")

declare -a adv=("pixelattack_spatialtransformation_deepfool")


## Iterate the string array using for loop
for a in ${adv[@]}; do
    python adv_detector_training.py --model resnet18 \
        --list $a \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 \
        --labelsmooth \
        --labelsmoothvalue 0.3 \
        --optimizer 'momentum' \
        --weight_decay 2e-4 \
        --batch-size 128 \
        --BNeval \
        --val 1000
    done
    

# bash ensemble_model.sh

# python adv_detector_training.py --model resnet18 \
#     --list pixelattack_spatialtransformation_deepfool \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 5 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval \
#     --val 1000
