## Introduction
This repository contains an improved version of the Junction Tree Variational Autoencoder (JTVAE) method. 

An autoencoder is typically composed of two neural networks: an encoder and a decoder. The encoder takes an input and compresses the input into some lower dimensionality referred to as the latent space. The decoder is trained to attempt to reconstruct the original input from the encoded vector.

Autoencoders are used in the drug discovery realm to encode molecules from their SMILES string to a vector of numbers then decode back to the original SMILES string. This allows us to attempt to use optimization techniques on the latent vector to create new molecules with optimized properties.

## Installation


## How we improved upon the original JTVAE method
While we haven't modified the underlying method of JTVAE, we have greatly improved the code and have added some new tools. Below is a description of some of the improvements we have made:

```
- Upgraded the code from Python 2 to Python 3
- Enabled batch training
- Tested the efficiency of running each of the 4 training steps on both CPU and GPU
- Modified code to enable GPU for the training step
- Integrated Pytorch's Distributed Data Parallel (DDP) method into the training step to parallelize model training across more than one GPU to accelerate training and reduce memory overloading
- Added parameters to enable the user to further define the latent space
- Provided batch scripts for each of the steps
```

## Getting started
There are four steps involved in training a JTVAE model:

```
1. Vocabulary Determination
2. Preprocessing
3. Training
4. Reconstruction Evaluation
```

## Issues
If you have any issues while using or installing this code, please create a new issue in the Issues tab of this github page. Someone will respond to the issue as soon as possible.

Feedback is always greatly appreciated! This code is still in development, changes will continue to be pushed and we have many plans for future development!

## Open-Source Development
If you want to suggest changes to the code base, please fork from the main branch then create a pull request. 
