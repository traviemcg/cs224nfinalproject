## Default Project++

This is a proejct for CS224N Winter 2020 at Stanford university. The goal is to better understand what is being learned by transformers through the use of logistic regression probes.

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

## Create conda environment with necessary packages, where pytorch may vary pending system but is at pytorch.org
conda create -n transformers python=3.7
conda activate transformers
pip install --upgrade pip
pip install --upgrade tensorflow
conda install pytorch torchvision -c pytorch


## (OPTIONAL) Make environment available in Jupyter, and install things needed for 'Transformers' notebooks
conda install -n transformers ipykernel
conda install -c anaconda jupyter
conda install -c conda-forge ipywidgets
conda update nbformat
python -m ipykernel install --user --name=transformers

## Install the 'Transformers' package
cd transformers-master
pip install .


```

### Using models

If you would like to use a tmux session (you would), one example is:
```
tmux new -s albert_xxlarge
tmux detach # disconnect using
tmux a -t albert_xxlarge # reconnect using
```

We are going to run eval with a Transformer community fine-tuned ALBERT [xlarge_v2](https://huggingface.co/ktrapeznikov/albert-xlarge-v2-squad-v2) and [xxlarge_v1](https://huggingface.co/ahotrod/albert_xxlargev1_squad2_512).
```
cd transformers-master/examples
conda activate transformers
export SQUAD_DIR=../../squad-master/data/
python run_squad.py --model_type albert --model_name_or_path ktrapeznikov/albert-xlarge-v2-squad-v2 --do_eval --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_xlarge_fine/
and/or
python run_squad.py --model_type albert --model_name_or_path ahotrod/albert_xxlargev1_squad2_512 --do_eval --do_lower_case --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 512 --doc_stride 128 --output_dir ./tmp/albert_xxlarge_fine/
```

albert_xlarge_v2
esults: {'exact': 84.33695294504771, 'f1': 87.35841153592796, 'total': 6078, 'HasAns_exact': 81.47766323024055, 'HasAns_f1': 87.78846230768734, 'HasAns_total': 2910, 'NoAns_exact': 86.96338383838383, 'NoAns_f1': 86.96338383838383, 'NoAns_total': 3168, 'best_exact': 84.33695294504771, 'best_exact_thresh': 0.0, 'best_f1': 87.35841153592791, 'best_f1_thresh': 0.0}

albert_xxlarge_v1
Results: {'exact': 85.32411977624218, 'f1': 88.83829560426527, 'total': 6078, 'HasAns_exact': 82.61168384879726, 'HasAns_f1': 89.95160160918354, 'HasAns_total': 2910, 'NoAns_exact': 87.81565656565657, 'NoAns_f1': 87.81565656565657, 'NoAns_total': 3168, 'best_exact': 85.32411977624218, 'best_exact_thresh': 0.0, 'best_f1': 88.83829560426533, 'best_f1_thresh': 0.0}

Then, for training our own ALBERT, let's train ALBERT v2 base on SQuAD v2.

albert_base
```
python run_squad.py --model_type albert --model_name_or_path albert-base-v2 --do_train --do_eval --do_lower_case --version_2_with_negative --train_file $SQUAD_DIR/train-v2.0.json --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_train_batch_size 8 --num_train_epochs 3 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_base_fine/
```