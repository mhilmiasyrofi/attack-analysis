# declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a train=("pixelattack" "squareattack" "spatialtransformation")

# declare -a train=("all")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# Iterate the string array using for loop
# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval_cifar10.py \
#             --train-adversarial $tr \
#             --test-adversarial $ts
#     done
# done

for ts in ${test[@]}; do
    python eval_cifar10.py \
        --train-adversarial combine \
        --list autopgd_pixelattack_spatialtransformation_squareattack \
        --test-adversarial $ts
done

# python eval_cifar10.py \
#     --train-adversarial squareattack \
#     --test-adversarial squareattack