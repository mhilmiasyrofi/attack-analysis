# declare -a train=("apgd" "autoattack" "bim" "cw" "deepfool" "ffgsm" "fgsm" "mifgsm" "pgd" "newtonfool" "jsma" "spatialtransformation" "squareattack" "tpgd")


# declare -a train=("original")

# declare -a test=("apgd" "autoattack" "bim" "cw" "deepfool" "ffgsm" "fgsm" "mifgsm" "pgd" "newtonfool" "jsma" "spatialtransformation" "squareattack" "tpgd")

# declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool" "elasticnet" "fgsm" "jsma"  "newtonfool" "pgd" "squareattack")

declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool" )
# declare -a train=("elasticnet" "fgsm" "jsma"  "newtonfool" "pgd" "squareattack")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "elasticnet" "jsma"  "newtonfool" "pgd" "squareattack")


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
            --batch-size 128 \
            --BNeval 
    done
done

# python eval.py --model resnet18 \
#     --train-adversarial deepfool \
#     --test-adversarial autoattack \
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
#     --batch-size 128 \
#     --BNeval 

python eval.py --model resnet18 \
    --train-adversarial original \
    --test-adversarial pgd \
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
    --batch-size 128 \
    --BNeval 