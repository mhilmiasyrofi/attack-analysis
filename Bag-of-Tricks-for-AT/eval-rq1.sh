## Used to run in different GPUs
# declare -a train=("original" "autoattack" "autopgd" )
# declare -a train=("bim" "cw" "fgsm")
# declare -a train=("pgd" "squareattack" "deepfool")
# declare -a train=("newtonfool" "pixelattack" "spatialtransformation")

# declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")

# declare -a train=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack" )


declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py --model resnet18 \
            --train-adversarial $tr \
            --test-adversarial $ts \
            --batch-size 128 \
            --model-dir ../trained_models/BagOfTricks/1000val/full/
    done
done

# python eval.py \
#     --model resnet18 \
#     --train-adversarial autoattack \
#     --test-adversarial autopgd \
#     --batch-size 128 \
#     --model-dir ../trained_models/BagOfTricks/1000val/full/
#     --val 1000 

