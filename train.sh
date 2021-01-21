# Declare an array of string with type
# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a adv=("autoattack" "autopgd" "bim")
# declare -a adv=("cw" "deepfool" "fgsm")
declare -a adv=("newtonfool" "pgd" "pixelattack")
# declare -a adv=("spatialtransformation" "squareattack")

# # Iterate the string array using for loop
for a in ${adv[@]}; do
    python adv_train_cifar.py \
        --attack $a 
done