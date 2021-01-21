python ensemble_model.py  \
    --model resnet18 \
    --centroids pixelattack_spatialtransformation_cw \
    --noise-predictor maxlr0.1_wd0.0001_ls0.3

# python ensemble_model.py  \
#     --model resnet18 \
#     --centroids pixelattack_spatialtransformation_squareattack_autopgd \
#     --noise-predictor maxlr0.05_wd0.0001_ls0.3