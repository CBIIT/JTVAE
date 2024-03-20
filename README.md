## Introduction
This repository contains an improved version of the Junction Tree Variational Autoencoder (JTVAE) method. 

An autoencoder is typically composed of two neural networks: an encoder and a decoder. The encoder takes an input and compresses the input into some lower dimensionality referred to as the latent space. The decoder is trained to attempt to reconstruct the original input from the encoded vector.

Autoencoders are used in the drug discovery realm to encode molecules from their SMILES string to a vector of numbers then decode back to the original SMILES string. This allows us to attempt to use optimization techniques on the latent vector to create new molecules with optimized properties.


## Installation
This software uses two separate conda environments: one for GPU and on for CPU. To install these two environments, it is recommended to use the provided environment specification files provided in this repository. Follow the below installations steps to create the two environments:

```
cd conda
conda create --name JTVAE-gpu-env --file gpu_env_file.txt
conda create --name JTVAE-cpu-env --file cpu_env_file.txt
```


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

The below sections further describe each of the steps, describes whether it uses a GPU or CPU, where to find the executable python file, and where to find an example slurm file for running that section on a Slurm managed system. It is recommended that for each project you want to use this for, to create a working directory outside of this repositories directory structure to preserve your data. Updating your local repository with remote version changes or creating any pull requests to make suggested improvements may overwrite your data if it is kept within your local copy of this repository.

### Vocabulary Determination
JTVAE requires generating a set of vocabulary which is specific to the compounds used for the train and test sets. The vocabulary is created by decomposing the list of SMILES strings into a list of unique functional units. This vocabulary is used in the encoding/decoding steps to describe the molecule and the pieces that make it up.

```
GPU/CPU: CPU
Executable location: JTVAE/CPU-P3/fast_jtnn/mol_tree.py
Example: LOGP-JTVAE-PAPER/Vocabulary/Slurm-Vocab-CPU
```

### Preprocessing
The training set of SMILES strings are split and saved into a specified number of pickled files. 

```
GPU/CPU: CPU
Executable location: JTVAE/CPU-P3/fast_molvae/preprocess.py
Example: LOGP-JTVAE-PAPER/Preprocess/Slurm-Preprocess-CPU
```

### Training
Autoencoder/Model is trained using the preprocessed files of SMILES strings.

```
GPU/CPU: GPU (one or more)
Executable location: JTVAE/GPU-P3/fast_molvae/vae_train_gpu.py
Example: LOGP-JTVAE-PAPER/Train/Slurm-Train-GPU
```

### Reconstruction Accuracy
The test set of SMILES strings (must not include any SMILES strings used in the training step) is put through the trained autoencoder by encoding the SMILES string into a latent vector then immediately decoded back to a SMILES string. The performance of the trained model is reported as the number and percent of SMILES strings that were correctly reconstructed.

```
GPU/CPU: CPU
Executable location: JTVAE/CPU-P3/fast_molvae/EDF.py
Example: LOGP-JTVAE-PAPER/Recon-Eval/Slurm-EDF-CPU
```


## Issues
If you have any issues while using or installing this code, please create a new issue in the Issues tab of this github page. Someone will respond to the issue as soon as possible.

Feedback is always greatly appreciated! This code is still in development, changes will continue to be pushed and we have many plans for future development!


## Open-Source Development
If you want to suggest changes to the code base, please fork from the main branch then create a pull request. 
