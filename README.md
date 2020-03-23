# Importance of Depth and Fine-tuning in BERT-models

Ths project started in [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) at Stanford University. The goal of this project is to better understand how transformer based pretrained natural language representations hierarchically represent information through the use of softmax regression probes. 

Our paper is here TODO. This repository will walk through all steps necessary to reproduce the results.

There are three major components to this respository:

- [squad2](squad2/) a data folder to save SQuAD 2.0 data splits provided by CS224N, further instruction in folder
- [transformers-master](https://github.com/huggingface/transformers) Huggingface's library providing easy access to many NLP models
- Our scripts for [training](train.py), [using](predict.py), and [evaluating results](evaluate.py) with [probes](probe.py)

## Setting up

```
## (OPTIONAL) General conda preperation
conda update conda
conda update --all
conda info # verify platform is 64 bit
curl https://sh.rustup.rs -sSf | sh # only on mac os
```

```
## Create conda environment with necessary packages, where pytorch may vary pending system but is at pytorch.org
conda create -n transformers python=3.7
conda activate transformers
pip3 install --upgrade pip tensorflow
conda install pytorch torchvision -c pytorch pandas
```

```
## Install the 'Transformers' package
cd transformers-master
pip3 install .
```

```
## Some useful tmux commands
tmux ls
tmux new -s session_name
tmux a -t session_name
tmux detach
```

## Using transformers models

### Community models

First let's try using a community trained fine-tuned ALBERT [xxlarge_v1](https://huggingface.co/ahotrod/albert_xxlargev1_squad2_512)

```
export SQUAD_DIR=../../squad2/
python3 run_squad.py --model_type albert --model_name_or_path ahotrod/albert_xxlargev1_squad2_512 --do_eval --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_xxlarge_fine/
```

```
Results: {'exact': 85.32411977624218, 'f1': 88.83829560426527, 'total': 6078, 'HasAns_exact': 82.61168384879726, 'HasAns_f1': 89.95160160918354, 'HasAns_total': 2910, 'NoAns_exact': 87.81565656565657, 'NoAns_f1': 87.81565656565657, 'NoAns_total': 3168, 'best_exact': 85.32411977624218, 'best_exact_thresh': 0.0, 'best_f1': 88.83829560426533, 'best_f1_thresh': 0.0}
```

### Training our own

Now, on to training our own ALBERT [base_v2](https://huggingface.co/twmkn9/albert-base-v2-squad2) on SQuAD v2.0 from a pretrained ALBERT base.
```
export SQUAD_DIR=../../squad2/
python3 run_squad.py --model_type albert --model_name_or_path albert-base-v2 --do_train --do_eval --overwrite_cache --do_lower_case --version_2_with_negative --train_file $SQUAD_DIR/train-v2.0.json --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_base_fine/
```

To use the model locally, where to use the community model one just needs to update the `--model_name_or_path` to twmkn9/albert-base-v2-squad2.
```
python3 run_squad.py --model_type albert --model_name_or_path ./tmp/albert_base_fine/ --do_eval --overwrite_cache --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_base_fine_test/
```

```
Results: {'exact': 78.71010200723923, 'f1': 81.89228117126069, 'total': 6078, 'HasAns_exact': 75.39518900343643, 'HasAns_f1': 82.04167868004215, 'HasAns_total': 2910, 'NoAns_exact': 81.7550505050505, 'NoAns_f1': 81.7550505050505, 'NoAns_total': 3168, 'best_exact': 78.72655478775913, 'best_exact_thresh': 0.0, 'best_f1': 81.90873395178066, 'best_f1_thresh': 0.0}
```

And, for comparison, train BERT [base-uncased](https://huggingface.co/bert-base-uncased).

```
python3 run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --overwrite_cache --do_lower_case --version_2_with_negative --train_file $SQUAD_DIR/train-v2.0.json --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/bert_base_fine/
```

```
Results: {'exact': 72.35932872655479, 'f1': 75.75355132564763, 'total': 6078, 'HasAns_exact': 74.29553264604812, 'HasAns_f1': 81.38490892002987, 'HasAns_total': 2910, 'NoAns_exact': 70.58080808080808, 'NoAns_f1': 70.58080808080808, 'NoAns_total': 3168, 'best_exact': 72.35932872655479, 'best_exact_thresh': 0.0, 'best_f1': 75.75355132564766, 'best_f1_thresh': 0.0}
```

## Probes

### Probe training

```
python3 train.py [model_prefix] [cpu/gpu] epochs
```

| Model             | Model Prefix                    |
|-------------------|---------------------------------|
| ALBERT Pretrained | albert-base-v2                  |
| ALBERT Fine-tuned | twmkn9/albert-base-v2-squad2    |
| BERT Pretrained   | bert-base-uncased               |
| BERT Fine-tuned   | twmkn9/bert-base-uncased-squad2 |

To train probes for each layer of ALBERT Pretrained on the cpu for 1 epoch (e.g. for debugging locally):
```
python3 train.py albert-base-v2 cpu 1
```

To train probes for each layer of ALBERT Fine-tuned on the gpu for 3 epoch (e.g. on a vm):
```
python3 train.py twmkn9/albert-base-v2-squad2 gpu 3
```

By default, probes will be saved for each epoch. If one is only interested in probes at a certain epoch, they can simply delete the unwanted directories.

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
