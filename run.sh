#!/bin/bash
 
declare -a attack=("autoattack" "apgd" "boundaryattack" "brendelbethge" "deepfool" "bim" "elasticnet" "pgd" "jsma" "shadowattack" "squareattack" "wasserstein")

# Iterate the string array using for loop
for a in ${attack[@]}; do
    python generate_adv_examples.py --attack $a
done