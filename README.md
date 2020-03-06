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

We are going to run eval with a Transformer [community fine-tuned ALBERT](https://huggingface.co/ktrapeznikov/albert-xlarge-v2-squad-v2)

```
cd transformers-master/examples
python run_squad.py --model_type albert --model_name_or_path ktrapeznikov/albert-xlarge-v2-squad-v2 --do_eval --do_lower_case --predict_file $SQUAD_DIR/dev-v2.0.json --max_seq_length 384 --doc_stride 128 --output_dir ./tmp/albert_xlarge_fine/
```