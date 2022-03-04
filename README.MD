# Beta-DARTS
This is the official implementation of $\beta$-DARTS: Beta-Decay Regularization for Differentiable Architecture Search (CVPR22).

## About
This code is based on the implementation of [DARTS](https://github.com/quark0/darts), [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects), [NAS-Bench-1Shot1](https://github.com/automl/nasbench-1shot1) and [SmoothDARTS](https://github.com/xiangning-chen/SmoothDARTS).

## Searching and Evaluation on NAS-Bench-201 Space:
Data Preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/AutoDL-Projects).

Searching and Evaluation: ```python ./nasbench201/train_search.py```

Note that we only use the training set when searching and retraining as DARTS. Please refer to line 135-139 of ./nasbench201/train_search.py to see the splitted training data.

## The Proposed Beta Decay Regularization
Please refer to ./optimizers/darts/architect.py to find our proposed loss as follows:

Line 55: ```def mlc_loss(self, arch_param):```

Please be noted that only “one line of code” （i.e. mlc loss）is added to make differentiable NAS methods much more Robustness and Generalization !!!

## Paper and Reference
[Paper](https://arxiv.org/pdf/2203.01665v1.pdf)
[Reference](https://arxiv.org/pdf/2203.01665v1.pdf)
