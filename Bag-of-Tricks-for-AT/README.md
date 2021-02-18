# Bag of Tricks for Adversarial Training
Adversarial training codes originated from [Bag of Tricks for Adversarial Training](https://github.com/P2333/Bag-of-Tricks-for-AT) 

## RQ1

Run a command for adversarial training
```
bash rq1-train.sh
```
This script will save the trained models inside `../trained_models/BagofTricks/1000val/full/`

Run evaluation using the trained models
```
bash rq1-eval.sh
```
This script will run evaluation on adversarial test-set then generate log on `../trained_models/BagofTricks/1000val/full/<train>/eval/best/<test>/`

Then perform: (1) clustering analysis using `rq1-clustering-analysis.ipynb`, (2) intracluster similarity and intercluster difference using `rq1-similarity-boxplot.ipynb`


## RQ3

Best performing model analysis: `rq3-analysis.ipynb`


## RQ4

Running command for adversarial training
```
bash rq4-train-sample.sh
```
The script will save the trained models inside `../trained_models/BagofTricks/1000val/25sample/`    

Run evaluation using the trained models
```
bash eval-rq1.sh
```
This script will .... generate log on `<fpath>`

Clustering analysis: `clustering-analysis-rq1.ipynb`
