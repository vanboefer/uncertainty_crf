Multi-class Uncertainty Cues Recognition
=======================================

# Description
The repo contains:

- code for creating an adjusted token-level version of the [Szeged Uncertainty Corpus](https://rgai.inf.u-szeged.hu/node/160) (Szarvas et al. 2012<sup>1</sup>);
- code for training and evaluating a CRF classifier, similar to the one trained by Szarvas et al. (2012).

For details, please refer to the [wiki]().

# Requirements
It is recommended to create a conda environment using the [environment.yml](environment.yml) file. This is done by running the command:

```
conda env create -f environment.yml
```
If you prefer to use ```pip```, you can find the names and versions of the required packages in [environment.yml](environment.yml).

# Usage
## Downloading the adjusted dataset
The adjusted version of the [Szeged Uncertainty Corpus](https://rgai.inf.u-szeged.hu/node/160) can be downloaded from [here](https://1drv.ms/u/s!AvPkt_QxBozXk7BiazucDqZkVxLo6g?e=IisuM6) in a form of a pickled pandas DataFrame (```szeged_fixed.pkl```, 172MB). For more information, refer to the ['Data' wiki page]().

## Using the CRF model on your own data
**NOTE**: My [HEDGEhog](https://github.com/vanboefer/hedgehog) repository contains a transformer-based model that performs the same multi-class classification task with ***better performance***. The CRF model in this repo was used as a baseline to evaluate HEDGEhog.

If you want to run the CRF model on your own data, use the [```predict.py```](code_ml/predict.py) script.

## Training your own CRF model
If you want to train your own CRF model, you can use the notebook [```train_multiclass_crf.ipynb```](code_ml/train_multiclass_crf.ipynb) as an example.

# References
<sup>1</sup> Szarvas, G., Vincze, V., Farkas, R., Móra, G., & Gurevych, I. (2012). Cross-genre and cross-domain detection of semantic uncertainty. *Computational Linguistics, 38*(2), 335-367.
