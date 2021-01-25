declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")

## Used to run in different GPUs
# declare -a train=("original" "autoattack" "autopgd" )
# declare -a train=("bim" "cw" "fgsm")
# declare -a train=("pgd" "squareattack" "deepfool")
# declare -a train=("newtonfool" "pixelattack" "spatialtransformation")
# declare -a train=("all")

for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py --model resnet18 \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --best-model \
            --lr-schedule piecewise \
            --norm l_inf \
            --epsilon 8 \
            --epochs 30 \
            --labelsmooth \
            --labelsmoothvalue 0.3 \
            --fname auto \
            --optimizer 'momentum' \
            --weight_decay 5e-4 \
            --batch-size 128 \
            --BNeval 
    done
done
