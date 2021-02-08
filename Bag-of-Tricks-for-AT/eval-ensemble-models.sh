## Used to run in different GPUs
declare -a train=("pixelattack_spatialtransformation_autoattack")

declare -a test=("spatialtransformation")

# declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# for tr in ${train[@]}; do
#     for ts in ${test[@]}; do
#         python eval.py --model resnet18 \
#             --train-adversarial $tr \
#             --test-adversarial $ts \
#             --batch-size 128 \
#             --model-dir ../trained_models/BagOfTricks/1000val/full/
#     done
# done

for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py --model resnet18 \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --batch-size 128 \
            --model-dir ../adv_detectors/1000val/full/
    done
done


# python eval.py \
#     --model resnet18 \
#     --train-adversarial autoattack \
#     --test-adversarial autopgd \
#     --batch-size 128 \
#     --model-dir ../trained_models/BagOfTricks/1000val/full/
#     --val 1000 

