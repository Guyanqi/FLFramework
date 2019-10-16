# Federated Learning Framework With Differential Privacy

## Description:
Federated Learning is a privacy preserving decentralized learning protocol introduced by Google. Multiple clients jointly learn a model without data centralization. Centralization is pushed from data space to parameter space: https://research.google.com/pubs/pub44822.html [1].
Differential privacy in deep learning is concerned with preserving privacy of individual data points: https://arxiv.org/abs/1607.00133 [2].
In this work we combine the notion of both by making federated learning differentially private. We focus on preserving privacy for the entire data set of a client. For more information, please refer to: https://arxiv.org/abs/1712.07557v2.

This code simulates a federated setting and enables federated learning with differential privacy. The privacy accountant used is from https://arxiv.org/abs/1607.00133 [2]. The files: accountant.py, utils.py, gaussian_moments.py are taken from: https://github.com/tensorflow/models/tree/master/research/differential_privacy

Note that the privacy agent is not completely set up yet (especially for more than 100 clients). It has to be specified manually or otherwise parameters 'm' and 'sigma' need to be specified.

## Requirements
- [TensorFlow 2.0](https://www.tensorflow.org/)
- [Python 3.6]