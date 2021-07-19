# A quantum algorithm for training wide and deep classical neural networks

## Abstract

Given the success of deep learning in classical machine learning, quantum algorithms for traditional neural network architectures quantum algorithms for traditional neural network architectures may provide one of the most promising settings for quantum machine learning. Considering a fully-connected feedforward neural network, we show that conditions amenable to classical trainability via gradient descent coincide with those necessary for efficiently solving quantum linear systems. We propose a quantum algorithm to approximately train a wide and deep neural network up to O(1/_n_) error for a training set of size _n_ by performing sparse matrix inversion in O(log _n_) time. To achieve an end-to-end exponential speedup over gradient descent, the data distribution must permit efficient state preparation and readout. We numerically demonstrate that the MNIST image dataset satisfies such conditions; moreover, the quantum algorithm matches the accuracy of the fully-connected network. Beyond the proven architecture, we provide empirical evidence for O(log _n_) training of a convolutional neural network with pooling.

Full paper: [arXiv link]

## Quick start

This repository implements a quantum algorithm to train a deep neural network via the neural tangent kernel framework, corresponding to the limit of infinite-width hidden layers. Two approximations to the neural tangent kernel (a sparsified and diagonal NTK) are shown to require logarithmic time for the MNIST image dataset. Implementations for both a fully-connected neural network and a convolutional neural network with pooling are available.

Code is written in python 3 using the [Neural Tangents](https://github.com/google/neural-tangents) library (tested in v0.3.6) and loads the MNIST dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview). It is highly recommended to use one or more GPUs due to the computational cost of evaluating the neural tangent kernel.

The neural network is trained in `training.py` and results are plotted in the `analysis.ipynb` notebook. By default, files will be saved to a directory `kernel_output`. Once dependencies are installed, the neural tangent kernel can be evaluated from the command line.

```shell
> mkdir kernel_output
> python training.py
```

To reduce runtime, the value of `trials` in `training.py` can be decreased; alternatively, for tighter error bars on performance, it is recommended to increase the quantity. Once `training.py` has finished evaluation, run `analysis.ipynb` to view the accuracy of MNIST classification and the scaling of relevant runtime quantities for the quantum algorithm (i.e. matrix sparsity and condition number, and the number of measurements for state preparation and readout).

## Citation

```
[bibtex]
```