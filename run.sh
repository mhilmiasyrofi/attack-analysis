#!/bin/bash
 

# declare -a attack=("autoattack" "apgd" "boundaryattack" "brendelbethge" "cw" "deepfool" "elasticnet" "fgm" "hopskipjump" "bim" "pgd" "pixelattack" "thresholdattack" "jsma" "shadowattack" "spatialtransformation" "squareattack" "universalperturbation" "wasserstein" "zoo")

# declare -a attack=("boundaryattack" "brendelbethge" "cw")

declare -a attack=("cw" "fgsm" "pgd" "squareattack" "bim" "newtonfool" "universalperturbation")

# cw fgsm bim pgd newtonfool squareattack universalperturbation

# Iterate the string array using for loop
for a in ${attack[@]}; do
    python generate_adv_examples.py --attack $a
done