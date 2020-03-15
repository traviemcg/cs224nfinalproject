## Default Project++

Ths project is for CS224N: Natural Language Processing with Deep Learning at Stanford University. The goal of this project is to better understand how transformer based pretrained natural language representations hierarchically represent information through the use of softmax regression probes. 

Our paper is here. This repository will walk through all steps necessary to reproduce the results.

There are three major components to this respository:

- [ALBERT-master](https://github.com/google-research/ALBERT) Google AI's original ALBERT implementation
- [squad-master](https://github.com/minggg/squad) CS224N's repository providing SQuAD 2.0 data and a BiDAF model
- [transformers-master](https://github.com/huggingface/transformers) Huggingface's library providing easy access to many NLP models

### Setting up

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
pip install --upgrade pip tensorflow
conda install pytorch torchvision -c pytorch pandas
```

```
## Install the 'Transformers' package
cd transformers-master
pip install .
```

```
## Some useful tmux commands
tmux ls
tmux new -s session_name
tmux a -t session_name
tmux detach
```

### Using models

#### Community models

First let's try using the community fine-tuned ALBERT [xlarge_v2](https://huggingface.co/ktrapeznikov/albert-xlarge-v2-squad-v2)

```
export SQUAD_DIR=../../squad-master/data/
python3 run_squad.py --model_type albert --model_name_or_path ktrapeznikov/albert-xlarge-v2-squad-v2 --do_eval --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_xlarge_fine/
```

```
Results: {'exact': 84.33695294504771, 'f1': 87.35841153592796, 'total': 6078, 'HasAns_exact': 81.47766323024055, 'HasAns_f1': 87.78846230768734, 'HasAns_total': 2910, 'NoAns_exact': 86.96338383838383, 'NoAns_f1': 86.96338383838383, 'NoAns_total': 3168, 'best_exact': 84.33695294504771, 'best_exact_thresh': 0.0, 'best_f1': 87.35841153592791, 'best_f1_thresh': 0.0}
```

And community fine-tuned ALBERT [xxlarge_v1](https://huggingface.co/ahotrod/albert_xxlargev1_squad2_512)

```
export SQUAD_DIR=../../squad-master/data/
python3 run_squad.py --model_type albert --model_name_or_path ahotrod/albert_xxlargev1_squad2_512 --do_eval --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_xxlarge_fine/
```

```
Results: {'exact': 85.32411977624218, 'f1': 88.83829560426527, 'total': 6078, 'HasAns_exact': 82.61168384879726, 'HasAns_f1': 89.95160160918354, 'HasAns_total': 2910, 'NoAns_exact': 87.81565656565657, 'NoAns_f1': 87.81565656565657, 'NoAns_total': 3168, 'best_exact': 85.32411977624218, 'best_exact_thresh': 0.0, 'best_f1': 88.83829560426533, 'best_f1_thresh': 0.0}
```

#### Training our own

We're training our own [base_v2](https://huggingface.co/twmkn9/albert-base-v2-squad2) on SQuAD v2.0 from the pretrained albert base.

```
export SQUAD_DIR=../../squad-master/data/
python3 run_squad.py --model_type albert --model_name_or_path twmkn9/albert-base-v2-squad2 --do_train --do_eval --do_lower_case --version_2_with_negative --train_file $SQUAD_DIR/train-v2.0.json --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_base_fine/ --overwrite_cache
```

To use the model locally

```
python3 run_squad.py --model_type albert --model_name_or_path ./tmp/albert_base_fine/ --do_eval --overwrite_cache --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_base_fine_test/
```

```
Results: {'exact': 78.71010200723923, 'f1': 81.89228117126069, 'total': 6078, 'HasAns_exact': 75.39518900343643, 'HasAns_f1': 82.04167868004215, 'HasAns_total': 2910, 'NoAns_exact': 81.7550505050505, 'NoAns_f1': 81.7550505050505, 'NoAns_total': 3168, 'best_exact': 78.72655478775913, 'best_exact_thresh': 0.0, 'best_f1': 81.90873395178066, 'best_f1_thresh': 0.0}
```

### Probe training

```
python3 qa_probes_iterative.py [pretrained/finetuned] [cpu/gpu] epochs
e.g. python3 qa_probes_iterative.py pretrained cpu 1
```

### Probe prediction

```
python3 eval_model.py [experiment/probes_dir] [exper/probes] device
```

Predictions can be done over a whole experiment directory
```
export EXPER_DIR=01_lr1e-5/
python3 eval_model.py $EXPER_DIR exper cpu
```

or one specific probe directory
```
export EPOCH_DIR=01_lr1e-5/fine_tuned_epoch_1
python3 eval_model.py $EPOCH_DIR/fine_tuned_probes probes cpu
```

### Evaluation

Evaluation can be done over a whole experiment directory
```
python3 evaluate.py experiment_directory exper
```

or one specific prediction directory
```
python3 evaluate.py preds_directory preds
```