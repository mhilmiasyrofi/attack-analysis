declare -a train=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")
declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")
# declare -a epochs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# declare -a epochs=(0 1 2 3 4)

# declare -a train=("autoattack")
# declare -a test=("autopgd")

# declare -a epochs=(4)
declare -a epochs=(0 1 2 3)

for ep in ${epochs[@]}; do
    for tr in ${train[@]}; do
        for ts in ${test[@]}; do
            python eval.py --model resnet18 \
                --train-adversarial $tr \
                --test-adversarial $ts \
                --model-epoch $ep \
                --train-adversarial $tr \
                --test-adversarial $ts \
                --batch-size 128 \
                --model-dir ../trained_models/BagOfTricks/1000val/50sample/ \
                --val 1000
        done
    done
done