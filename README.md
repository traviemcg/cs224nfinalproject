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
    - [Training our own](#training-our-own)
3. [Probes](#probes)
    - [Model prefixes](#model-prefixes)
    - [Probe training](#probe-training)
    - [Probe prediction](#probe-prediction)
    - [Probe evaluation](#probe-evaluation)
4. [Reproducing results](#reproducing-results)

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

### Training our own

Now, on to training [our own ALBERT](https://huggingface.co/twmkn9/albert-base-v2-squad2) on SQuAD v2.0 from [a pretrained ALBERT base v2](https://huggingface.co/albert-base-v2).
```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type albert 
    --model_name_or_path albert-base-v2 
    --do_train 
    --do_eval 
    --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --train_file $SQUAD_DIR/train-v2.0.json 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5 
    -max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/albert_fine/
```

To use the trained model again locally:
```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type albert 
    --model_name_or_path ./tmp/albert_fine/ 
    --do_eval --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5
    --max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/albert_fine_dev/
```

We'll also train [our own BERT](https://huggingface.co/twmkn9/bert-base-uncased-squad2) from an [uncased BERT base](https://huggingface.co/bert-base-uncased).

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type bert 
    --model_name_or_path bert-base-uncased 
    --do_train 
    --do_eval 
    --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --train_file $SQUAD_DIR/train-v2.0.json 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5 
    --max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/bert_fine/
```

[Our own DistilBERT](https://huggingface.co/twmkn9/distilbert-base-uncased-squad2) from an [uncased DistilBERT base](https://huggingface.co/distilbert-base-uncased).

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type distilbert 
    --model_name_or_path distilbert-base-uncased
    --do_train 
    --do_eval 
    --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --train_file $SQUAD_DIR/train-v2.0.json 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5 
    --max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/distilbert_fine/
```

[Our own DistilRoberta]() from an [DistilRoberta base](https://huggingface.co/distilroberta-base).

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type robberta 
    --model_name_or_path distilroberta-base 
    --do_train 
    --do_eval 
    --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --train_file $SQUAD_DIR/train-v2.0.json 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5 
    --max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/distilrobberta_fine/
```

| Model                 | Exact | F1    | Exact Has Ans | F1 Has Ans | Exact No Ans | F1 No Ans |
|-----------------------|-------|-------|---------------|------------|--------------|-----------|
| BERT Fine-tuned       | 72.36 | 75.75 | 74.30         | 81.38      | 70.58        | 70.58     |
| ALBERT Fine-tuned     | 78.71 | 81.89 | 75.40         | 82.04      | 81.76        | 81.76     |
| DistilBERT Fine-tuned | 64.89 | 68.18 | 69.76         | 76.63      | 60.42        | 60.42     |

Results: {'exact': 64.88976637051661, 'f1': 68.1776176526635, 'total': 6078, 'HasAns_exact': 69.7594501718213, 'HasAns_f1': 76.62665295288285, 'HasAns_total': 2910, 'NoAns_exact': 60.416666666666664, 'NoAns_f1': 60.416666666666664, 'NoAns_total': 3168, 'best_exact': 64.88976637051661, 'best_exact_thresh': 0.0, 'best_f1': 68.17761765266337, 'best_f1_thresh': 0.0}

## Probes

### Model prefixes

At various times, we will want to reference models by their prefix in the transformers library, so a table is provided.

| Model                    | Model Prefix                         |
|-------------------------|---------------------------------------|
| ALBERT Pretrained       | albert-base-v2                        |
| ALBERT Fine-tuned       | twmkn9/albert-base-v2-squad2          |
| BERT Pretrained         | bert-base-uncased                     |
| BERT Fine-tuned         | twmkn9/bert-base-uncased-squad2       |
| DistilBERT Fine-tuned   | distilbert-base-uncased               |
| BERT Fine-tuned         | twmkn9/distilbert-base-uncased-squad2 |


### Probe training

```
python3 train.py [model_prefix] [cpu/gpu] epochs
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

## Reproducing results

FORTHCOMING