## Default Project++

This is a proejct for CS224N Winter 2020 at Stanford university. The goal is to better understand what is being learned by transformers through the use of logistic regression probes.

Our paper is here. This repository will walk through all steps necessary to reproduce the results.

There are three major components to this respository:

- [ALBERT-master](https://github.com/google-research/ALBERT) Google AI's original ALBERT implementation
- [squad-master](https://github.com/minggg/squad) CS224N's repository providing SQuAD 2.0 data and a BiDAF model
- [transformers-master](https://github.com/huggingface/transformers) Huggingface's library providing easy access to many NLP models

### Setting up

```
conda update conda
conda update --all
conda info # verify platform is 64 bit
curl https://sh.rustup.rs -sSf | sh # mac os

conda create -n transformers python=3.7
conda activate transformers
pip install --upgrade pip
pip install --upgrade tensorflow
conda install pytorch torchvision -c pytorch

cd transformers-master
pip install .
```
