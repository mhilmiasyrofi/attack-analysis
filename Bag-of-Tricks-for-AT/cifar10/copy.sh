declare -a attack=("original" "autoattack" "autopgd" "bim" "cw" "fgsm" "pgd" "squareattack" "deepfool" "newtonfool" "pixelattack" "spatialtransformation")

if [ ! -d "csv" ] 
then
    echo "Making directory csv"
    mkdir csv
fi

for a in ${attack[@]}; do
    if [ ! -d csv/$a ] 
    then
        echo "Making directory csv/$a"
        mkdir csv/$a
    fi
done

for a in ${attack[@]}; do
    cp -r trained_models/*$a*/eval/ csv/$a
done
