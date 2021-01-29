declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a train=("autoattack" "autopgd" "bim" )
# declare -a train=("cw" "deepfool" "fgsm")
# declare -a train=("newtonfool" "pgd")
# declare -a train=("pixelattack" "squareattack" "spatialtransformation")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


## Iterate the string array using for loop
# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval.py \
#             --train-adversarial $tr \
#             --test-adversarial $ts \
#             --model-dir ../trained_models/AT/1000val/full/
#     done
# done

for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --model-dir ../trained_models/AT/1000val/full/ \
            --val 1000
    done
done

# python eval.py \
#     --train-adversarial autopgd \
#     --test-adversarial fgsm \
#     --model-dir ../trained_models/AT/1000val/full/




