# declare -a train=("original" "autoattack" "autopgd" )
# declare -a train=("bim" "cw" "fgsm")
# declare -a train=("pgd" "squareattack" "deepfool")
# declare -a train=("newtonfool" "pixelattack" "spatialtransformation")
# declare -a train=("all")

# declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")

# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval.py --model resnet18 \
#             --train-adversarial $tr \
#             --test-adversarial $ts \
#             --best-model \
#             --lr-schedule piecewise \
#             --norm l_inf \
#             --epsilon 8 \
#             --epochs 30 \
#             --labelsmooth \
#             --labelsmoothvalue 0.3 \
#             --fname auto \
#             --optimizer 'momentum' \
#             --weight_decay 5e-4 \
#             --batch-size 128 \
#             --BNeval 
#     done
# done

declare -a train=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")
declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")
# declare -a epochs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
declare -a epochs=(0 1 2 3 4 5 6 7 8 9)

for ep in ${epochs[@]}; do
    for tr in ${train[@]}; do
        for ts in ${test[@]}; do
            python eval.py --model resnet18 \
                --train-adversarial $tr \
                --test-adversarial $ts \
                --model-epoch $ep \
                --sample 100 \
                --fname ../../trained_models/ \
                --lr-schedule piecewise \
                --norm l_inf \
                --epsilon 8 \
                --labelsmooth \
                --labelsmoothvalue 0.3 \
                --optimizer 'momentum' \
                --weight_decay 5e-4 \
                --batch-size 128 \
                --BNeval 
        done
    done
done



# python eval.py --model resnet18 \
#     --train-adversarial autoattack \
#     --test-adversarial autopgd \
#     --model-epoch 0 \
#     --sample 100 \
#     --fname ../../trained_models/ \
#     --lr-schedule piecewise \
#     --norm l_inf \
#     --epsilon 8 \
#     --labelsmooth \
#     --labelsmoothvalue 0.3 \
#     --optimizer 'momentum' \
#     --weight_decay 5e-4 \
#     --batch-size 128 \
#     --BNeval 

# declare -a test=("autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")

# for ts in ${test[@]}; do
#     python eval.py --model resnet18 \
#         --train-adversarial combine \
#         --list pixelattack_spatialtransformation_cw_autopgd \
#         --test-adversarial $ts \
#         --best-model \
#         --lr-schedule piecewise \
#         --norm l_inf \
#         --epsilon 8 \
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
#     --list autoattack_pixelattack_spatialtransformation \
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