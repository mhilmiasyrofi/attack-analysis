declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


## Iterate the string array using for loop
# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval_cifar10.py \
#             --train-adversarial $tr \
#             --test-adversarial $ts \
#             --model-dir ../trained_models/AWP/1000val/full/
#     done
# done

for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval_cifar10.py \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --model-dir ../trained_models/AWP/1000val/full/ \
            --val 1000
    done
done