# Bag of Tricks for Adversarial Training
Adversarial training codes originated from [Bag of Tricks for Adversarial Training](https://github.com/P2333/Bag-of-Tricks-for-AT) 

## RQ1

Running command for adversarial training
```
bash train-rq1.sh
```
This script will generate trained models that is saved at `../trained_models/BagofTricks/1000val/full/`

Run evaluation using the trained models
```
bash eval-rq1.sh
```
This script will run evaluation on adversarial test-set then generate log on `../trained_models/BagofTricks/1000val/full/<train>/eval/best/<test>/`

Then perform clustering on `clustering-analysis-rq1.ipynb`


## RQ3

Best performing model analysis: `best-performing-model-analysis.ipynb`


## RQ4

Running command for adversarial training
```
bash train-rq1.sh
```
This script will generate trained models that is saved at `<fpath>`

Run evaluation using the trained models
```
bash eval-rq1.sh
```
This script will .... generate log on `<fpath>`

Clustering analysis: `clustering-analysis-rq1.ipynb`
