# declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "fgsm")
declare -a train=("pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")


declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")


for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py --model resnet18 \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --best-model \
            --lr-schedule piecewise \
            --norm l_inf \
            --epsilon 8 \
            --epochs 110 \
            --attack-iters 10 \
            --pgd-alpha 2 \
            --fname auto \
            --optimizer 'momentum' \
            --weight_decay 5e-4 \
            --batch-size 256 \
            --BNeval 
    done
done

# python eval.py --model resnet18 \
#     --train-adversarial spatialtransformation \
#     --test-adversarial spatialtransformation \
#     --best-model \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --attack-iters 10 \
#     --pgd-alpha 2 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 256 \
#     --BNeval 

# python eval.py --model resnet18 \
#     --train-adversarial original \
#     --test-adversarial pgd \
#     --best-model \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --attack-iters 10 \
#     --pgd-alpha 2 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 256 \
#     --BNeval 