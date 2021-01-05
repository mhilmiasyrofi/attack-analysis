#!/bin/bash
 
# docker run -it --rm --name gpu0-bot -v /home/mhilmiasyrofi/Documents/Bag-of-Tricks-for-AT/:/workspace/Bag-of-Tricks-for-AT/ --gpus '"device=0"' mhilmiasyrofi/advtraining
 
# Declare an array of string with type
# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a adv=("autoattack" "autopgd" "bim")
# declare -a adv=("cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack")
# declare -a adv=("spatialtransformation" "squareattack")

# # Iterate the string array using for loop
# for a in ${adv[@]}; do
#     python adversarial_training.py --model resnet18 \
#         --attack $a \
#         --lr-schedule piecewise \
#         --norm l_inf \
#         --epsilon 8 \
#         --epochs 110 \
#         --labelsmooth \
#         --labelsmoothvalue 0.3 \
#         --fname auto \
#         --optimizer 'momentum' \
#         --weight_decay 5e-4 \
#         --batch-size 128 \
#         --BNeval 
# done

# declare -a adv=("autoattack" "autopgd" "bim" "cw")
# # declare -a adv=("deepfool" "fgsm" "newtonfool" "pgd")
# # declare -a adv=("pixelattack" "spatialtransformation" "squareattack")

# Iterate the string array using for loop
# for a in ${adv[@]}; do
#     python adversarial_training.py --model resnet18 \
#         --attack $a \
#         --sample 50 \
#         --lr-schedule piecewise \
#         --norm l_inf \
#         --epsilon 8 \
#         --epochs 30 \
#         --labelsmooth \
#         --labelsmoothvalue 0.3 \
#         --fname auto \
#         --optimizer 'momentum' \
#         --weight_decay 5e-4 \
#         --batch-size 128 \
#         --BNeval 
# done

# python adversarial_training.py --model resnet18 \
#     --attack pgd \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 20 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 256 \
#     --BNeval 


# python adversarial_training.py --model resnet18 \
#     --attack combine \
#     --list pixelattack_spatialtransformation_autoatack_autopgd \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval
    
# python adversarial_training.py --model resnet18 \
#     --attack combine \
#     --list pixelattack_spatialtransformation_cw_autopgd \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval

# python adversarial_training.py --model resnet18 \
#     --attack all \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval
