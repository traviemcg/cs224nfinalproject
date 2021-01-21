# Importance of Depth and Fine-tuning in BERT-models

Ths project started in [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) at Stanford University. The goal of this project is to better understand how transformer based pretrained natural language representations hierarchically represent information through the use of softmax regression probes. 

Our paper is here FORTHCOMING. This repository will walk through all steps necessary to reproduce the results.

There are three major components to this respository:

- [squad2](squad2/) a data folder to save [SQuAD 2.0 data splits](https://github.com/minggg/squad/tree/master/data) provided by Stanford's CS224N
- [transformers-master](https://github.com/huggingface/transformers) Huggingface's library providing easy access to many NLP models
- Our scripts for [training](train.py), [using](predict.py), and [evaluating results](evaluate.py) with [probes](probe.py)

# Table Contents 

1. [Setting up](#setting-up)
2. [Using transformers models](#using-transformers-models)
    - [Community models](#community-models)
    - [Our models](#training-our-own)
3. [Probes](#probes)
    - [Probe training](#probe-training)
    - [Probe prediction](#probe-prediction)
    - [Probe evaluation](#probe-evaluation)
4. [Results](#results)

## Setting up

(OPTIONAL) General conda preperation:
```
conda update conda
conda update --all
conda info # verify platform is 64 bit
curl https://sh.rustup.rs -sSf | sh # only on mac os
```

Create a conda environment with the necessary packages, where pytorch may vary pending system but is at [pytorch.org](pytorch.org).
```
conda create -n transformers python=3.7
conda activate transformers
pip3 install --upgrade pip tensorflow
conda install pytorch torchvision -c pytorch pandas
```

Then install the revision of the 'Transformers' package associated with this library.
```
cd transformers-master
pip3 install .
```

(OPTIONAL) Some useful tmux commands:
```
tmux ls
tmux new -s session_name
tmux a -t session_name
tmux detach
```

## Using transformers models

### Community models

First, be sure you have downloaded `train-v2.0.json` and `dev-v2.0.json` to [squad2](squad2) as specified in the [README](squad2/README.md) Then, move into the transformer-master directory.

```
cd transformers-master/examples
```

First, use a community trained ALBERT [xxlarge_v1](https://huggingface.co/ahotrod/albert_xxlargev1_squad2_512) fine-tuned

```
export SQUAD_DIR=../../squad2/
python3 run_squad.py 
    --model_type albert 
    --model_name_or_path ahotrod/albert_xxlargev1_squad2_512 
    --do_eval 
    --do_lower_case 
    --version_2_with_negative 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --max_seq_length 384 --doc_stride 128 
    --output_dir ./tmp/albert_xxlarge_fine/
```

| Model                 | Exact | F1    | Exact Has Ans | F1 Has Ans | Exact No Ans | F1 No Ans |
|-----------------------|-------|-------|---------------|------------|--------------|-----------|
| ALBERT v1 XXLarge     | 85.32 | 88.84 | 82.61         | 89.95      | 87.82        | 87.82     |

### Our models

At various times, we will want to reference models by their prefix in the transformers library, so a table is provided. The pretrained models were created and shared by the Huggingface team (creators of the transformers library), while the fine-tuned models were trained and shared by us. The exact Python commands used to train each model, along with more detailed model performance, is included on each of the linked model cards.

| Model                    | Model Prefix                                                                                          |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| ALBERT Pretrained        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                               |
| ALBERT Fine-tuned        | [twmkn9/albert-base-v2-squad2](https://huggingface.co/twmkn9/albert-base-v2-squad2)                   |
| BERT Pretrained          | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                         |
| BERT Fine-tuned          | [twmkn9/bert-base-uncased-squad2](https://huggingface.co/twmkn9/bert-base-uncased-squad2)             |
| DistilBERT Pretrained    | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                             |
| DistilBERT Fine-tuned    | [twmkn9/distilbert-base-uncased-squad2](https://huggingface.co/twmkn9/distilbert-base-uncased-squad2) |
| DistilRoberta Pretrained | [distilroberta-base](https://huggingface.co/distilroberta-base)                                       |
| DistilRoberta Fine-tuned | [twmkn9/distilroberta-base-squad2](https://huggingface.co/twmkn9/distilroberta-base-squad2)           |

| Model                    | Exact | F1    | Exact Has Ans | F1 Has Ans | Exact No Ans | F1 No Ans |
|--------------------------|-------|-------|---------------|------------|--------------|-----------|
| BERT Fine-tuned          | 72.36 | 75.75 | 74.30         | 81.38      | 70.58        | 70.58     |
| ALBERT Fine-tuned        | 78.71 | 81.89 | 75.40         | 82.04      | 81.76        | 81.76     |
| DistilBERT Fine-tuned    | 64.89 | 68.18 | 69.76         | 76.63      | 60.42        | 60.42     |
| DistilRoberta Fine-tuned | 70.93 | 74.60 | 67.63         | 75.30      | 73.96        | 73.96     |

## Probes

### Probe training

```
python3 train.py [model_prefix] [cpu/gpu] [epochs]
```

To train probes for each layer of ALBERT Pretrained on the cpu for 1 epoch (e.g. for debugging locally):
```
python3 train.py albert-base-v2 cpu 1
```

To train probes for each layer of ALBERT Fine-tuned on the gpu for 3 epoch (e.g. on a vm):
```
python3 train.py twmkn9/albert-base-v2-squad2 gpu 3
```

By default, probes will be saved for each epoch. If one is only interested in probes at a certain epoch, they can simply delete the unwanted intermediate epoch directories.

### Probe prediction

```
python3 predict.py [model_prefix] [cpu/gpu]
```

To make predictions for probes at each layer and each epoch for BERT Pretrained on the cpu:
```
python3 predict.py bert-base-uncased cpu
```

### Probe evaluation

```
python3 evaluate.py [model_prefix]
```

To evaluate predictions for probes at each layer and each epoch for BERT Fine-tuned:
```
python3 evaluate.py twmkn9/bert-base-uncased-squad2
```

## Results

For the final results of our project, please check out the [paper](CS224N_Final_Report.pdf)! 