# declare -a train=("original" "autoattack" "autopgd" )
# declare -a train=("bim" "cw" "fgsm")
# declare -a train=("pgd" "squareattack" "deepfool")
declare -a train=("newtonfool" "pixelattack" "spatialtransformation")
# declare -a train=("all")

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
            --epochs 10 \
            --labelsmooth \
            --labelsmoothvalue 0.3 \
            --fname auto \
            --optimizer 'momentum' \
            --weight_decay 5e-4 \
            --batch-size 128 \
            --BNeval 
    done
done

# declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")

# for ts in ${test[@]}; do
#     python eval.py --model resnet18 \
#         --train-adversarial combine \
#         --list newtonfool_pixelattack_spatialtransformation_squareattack \
#         --test-adversarial $ts \
#         --best-model \
#         --lr-schedule piecewise \
#         --norm l_inf \
#         --epsilon 8 \
#         --epochs 110 \
#         --labelsmooth \
#         --labelsmoothvalue 0.3 \
#         --fname auto \
#         --optimizer 'momentum' \
#         --weight_decay 5e-4 \
#         --batch-size 128 \
#         --BNeval 
# done


# python eval.py --model resnet18 \
#     --train-adversarial spatialtransformation \
#     --test-adversarial spatialtransformation \
#     --best-model \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 256 \
#     --BNeval 

# python eval.py --model resnet18 \
#     --train-adversarial combine \
#     --list newtonfool_pixelattack_spatialtransformation_squareattack \
#     --test-adversarial pgd \
#     --best-model \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --epochs 110 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --fname auto \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval 