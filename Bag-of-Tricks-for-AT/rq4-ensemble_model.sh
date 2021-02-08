declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a test=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack" )

# declare -a centroids=("pixelattack_spatialtransformation_autoattack" "pixelattack_spatialtransformation_bim" "pixelattack_spatialtransformation_deepfool")

# declare -a centroids=("pixelattack_spatialtransformation_autoattack" "pixelattack_spatialtransformation_bim") 

# declare -a centroids=("pixelattack_spatialtransformation_bim" "pixelattack_spatialtransformation_deepfool")
declare -a centroids=("pixelattack_spatialtransformation_deepfool")



for c in ${centroids[@]}; do
    for ts in ${test[@]}; do
        python ensemble_model.py  \
            --model resnet18 \
            --centroids $c \
            --test-adversarial $ts
        done
    done

# python ensemble_model.py  \
#     --model resnet18 \
#     --centroids pixelattack_spatialtransformation_autoattack \
#     --test-adversarial pgd