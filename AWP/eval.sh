# declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a train=("pixelattack" "squareattack" "spatialtransformation")

# declare -a train=("all")

# declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# Iterate the string array using for loop
# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval_cifar10.py \
#             --train-adversarial $tr \
#             --test-adversarial $ts \
#     done
# done


declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")
declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")
declare -a epochs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# declare -a train=("autoattack")
# declare -a test=("autopgd")


dddd## Iterate the string array using for loop
# for ep in ${epochs[@]}; do
#     for tr in ${train[@]}; do
#         for ts in ${test[@]}; do
#             python eval_cifar10.py \
#                 --train-adversarial $tr \
#                 --test-adversarial $ts \
#                 --model-epoch $ep \
#                 --fname ../../trained_models/default/
#         done
#     done
# done

# for ts in ${test[@]}; do
#     python eval_cifar10.py \
#         --train-adversarial combine \
#         --list autopgd_pixelattack_spatialtransformation_squareattack \
#         --test-adversarial $ts
# done

# python eval_cifar10.py \
#     --train-adversarial squareattack \
#     --test-adversarial squareattack