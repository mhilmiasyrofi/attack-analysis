#!/bin/bash

# declare -a attack=("autoattack" "autopgd" "bim")

# declare -a attack=("cw" "deepfool" "fgsm")

# declare -a attack=("newtonfool" "pgd" "pixelattack" )

declare -a attack=("spatialtransformation" "squareattack")

# declare -a attack=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# Iterate the string array using for loop
for a in ${attack[@]}; do
    python oat_generate_adv_examples.py --attack $a
done