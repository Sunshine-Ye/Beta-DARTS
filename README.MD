# Paper Title: Beta-DARTS

## Datasets
You can download CIFAR-10, CIFAR-100 and SVHN datasets by automatically download process using torchvision.
Note that : 1) instructions for acquiring PTB is [here](https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh). 
2) to use NAS-Bench-1Shot1 to monitor the anytime test accuracy of NAS algorithms, please download this tfrecord:
[https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord).
Then insert its path to ```nasbench_analysis/eval_darts_one_shot_model_in_nasbench.py```

## Architecture Training and Evaluation

```Beta-DARTS: python ./nasbench201/train_search.py```



