# Kernel-Based Approaches for Sequence Modeling: Connections to Neural Methods

This repository contains the code and poster for the NeurIPS 2019 paper [Kernel-Based Approaches for Sequence Modeling: Connections to Neural Methods](https://arxiv.org/abs/1910.04233).

If you find our work useful for your research, please consider citing:
```
@article{kernels2rnns,
  title={{Kernel-Based Approaches for Sequence Modeling: Connections to Neural Methods}},
  author={Liang, Kevin J and Wang, Guoyin and Li, Yitong and Henao, Ricardo and Carin, Lawrence},
  journal={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Document Classification
This repository contains a reimplementation of the code used for document classification.
These changes make the code clearer, but may also result in slightly different results than those reported in the paper.

### Pre-requisites
*Software:* The document classification code requires Python 3 and TensorFlow 1.[x] (much of the development was done with TF 1.9).
See [here](https://github.com/duke-mlss/PlusDataScience/blob/61f7e8b7d1679b38d199cbdb9933e26e0b21acc8/1A_TensorFlow_Installation.ipynb) for example installation instructions.

*Hardware:* While not technically required, you'll probably want to use a CUDA-enabled GPU. 
We used an NVIDIA Titan X for our experiments.

### Datasets
We consider the following datasets: AGnews, DBPedia, Yahoo!, and Yelp Full. For convenience, we provide pre-processed versions of all datasets. Data are prepared in pickle format. Each `.p` file has the same fields in same order: `train text`, `val text`, `test text`, `train label`, `val label`, `test label`, `dictionary` and `reverse dictionary`.

Datasets can be downloaded [here](https://drive.google.com/open?id=1QmZfoKSgZl8UMN8XenAYqHaRzbW5QA26). Place the downloaded data in a directory named `[$ROOT]/Data/`. Each dataset has two files: the tokenized data and the corresponding pre-trained Glove embeddings.

### Training a Classifier
We provide an example [script](https://github.com/kevinjliang/kernels2rnns/blob/master/run_model.sh) that trains a classifier on each of the 4 datasets considered in the paper.
For example,
```Shell
    bash run_model.sh 0 rkm_lstm 1
```
will train a 1-gram RKM-LSTM classifier on AGnews, DBPedia, Yahoo!, and Yelp Full, using the first CUDA-visible GPU.
The full list of flags can be found in [`main.py`](https://github.com/kevinjliang/kernels2rnns/blob/master/main.py).
