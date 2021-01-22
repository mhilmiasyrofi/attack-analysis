declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# Iterate the string array using for loop
for a in ${adv[@]}; do
    python adv_train_cifar10.py \
        --attack $a \
        --batch-size 128 \
        --chkpt-iters 1 \
        --epochs 20
done


# python adv_train_cifar10.py \
#         --attack autoattack \
#         --batch-size 128 \
#         --chkpt-iters 1 \
#         --sample 50 \
#         --epochs 6jn