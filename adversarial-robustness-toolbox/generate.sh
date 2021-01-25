#!/bin/bash
 
## Old Version
# declare -a attack=("autoattack" "apgd" "boundaryattack" "brendelbethge" "cw" "deepfool" "elasticnet" "fgm" "hopskipjump" "bim" "pgd" "pixelattack" "thresholdattack" "jsma" "spatialtransformation" "squareattack" "universalperturbation" "wasserstein" "zoo")

declare -a attack=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a attack=("autoattack" "autopgd" "bim")
# declare -a attack=("cw" "deepfool" "fgsm")
# declare -a attack=("newtonfool" "pgd" "pixelattack")
# declare -a attack=("spatialtransformation" "squareattack")


# Iterate the string array using for loop
for a in ${attack[@]}; do
    python generate_adv_examples.py --attack $a
done