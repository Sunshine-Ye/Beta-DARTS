# Beta-DARTS

## Datasets
This code is based on the implementation of [DARTS](https://github.com/quark0/darts), [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects), [NAS-Bench-1Shot1](https://github.com/automl/nasbench-1shot1) and [SmoothDARTS](https://github.com/xiangning-chen/SmoothDARTS).

## Architecture Training and Evaluation

```Beta-DARTS: python ./nasbench201/train_search.py```

## The Proposed Loss
Please refer to ./optimizers/darts/architect.py to find our proposed loss as follows:

```def mlc_loss(self, arch_param):```

Please be noted that our codes are based on DARTS an SmoothDARTS, and only “one line of code” （i.e. mlc loss）is added to make differentiable NAS methods much more robust and generalization.


