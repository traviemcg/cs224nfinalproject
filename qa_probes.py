import numpy as np
from transformers import *
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadV2Processor
import sys
from SoftmaxRegression import MultiSoftmaxRegression
import os
import pandas as pd



def train_probes(model_prefix,
                 data_dir,
                 filename,
                 probe_dir,
                 layers = 12,
                 epoches = 1,
                 hidden_dim = 768,
                 max_seq_length = 384,
                 batch_size = 4,
                 device = 'cpu'):
    '''
       Trains softmax probe corresponding to each layer of Albert

    '''

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir = data_dir, filename = filename)

    # examples = examples[:8]

    # Extract features
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset="pt",
        threads=1,
    )

    # Initialize ALBERT model
    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Initialize probes
    print("Initializing probes")
    probes = []
    for i in range(layers):
        probes.append(MultiSoftmaxRegression(hidden_dim))

    # Training epoches
    for epoch in range(epoches):

        print("TRAINING EPOCH: {}".format(epoch))

        # Initialize data loaders
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

        # Training batches
        for batch in tqdm(train_dataloader, desc = "Iteration"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            
            with torch.no_grad():

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                # Albert forward pass
                outputs = model(**inputs)
                attention_hidden_states = outputs[3][1:]

                # Update probes
                for j in range(batch[7].shape[0]):
                    start = batch[3][j].clone().unsqueeze(0).to(device)
                    end  = batch[4][j].clone().unsqueeze(0).to(device)

                    # Train probes
                    for i, p in enumerate(probes):
                        p.train(attention_hidden_states[i][j].unsqueeze(0), start, end, device)

    # Save probes
    for i, p in enumerate(probes):
        p.save(probe_dir, i)

def evaluate_probes(model_prefix,
                    data_dir,
                    filename,
                    probe_dir,
                    pred_dir,
                    layers = 12,
                    hidden_dim = 768,
                    max_seq_length = 384,
                    device = 'cpu'):

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir = data_dir, filename = filename)

    examples = examples[:20]

    # Extract features
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    # Initialize ALBERT model
    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    # Initialize data loaders
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = 4)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Load probes
    print("Loading probes")
    probes = []
    for i in range(layers):
        p = MultiSoftmaxRegression(hidden_dim)
        p.load(probe_dir, i)
        probes.append(p)

    # Extract IDs
    print("Extracting IDs")
    n = len(examples)
    q_ids = []
    for i in range(n):
        q_ids.append(examples[i].qas_id)

    # Initialize predictions
    predictions = []
    for i in range(layers):
        pred = pd.DataFrame()
        pred['Id'] = q_ids
        pred['Predicted'] = [""] * len(examples)
        predictions.append(pred)

    # Evaluation batches
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

            # Albert forward pass
            idx = batch[3]
            outputs = model(**inputs)
            attention_hidden_states = outputs[2][1:]

            # Compute prediction
            for j, index in enumerate(idx):
                index = int(index.item())
                if index >= n:
                    break
                for i, p in enumerate(probes):

                    # Extract predicted indicies
                    start_idx, end_idx = p.predict(attention_hidden_states[i][j].unsqueeze(0), device)
                    start_idx = int(start_idx[0])
                    end_idx = int(end_idx[0])

                    # Extract predicted answer
                    tokens = tokenizer.convert_ids_to_tokens(batch[0][j])
                    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

                    # No answer
                    if answer == '[CLS]':
                        answer = ''

                    # Populate output
                    predictions[i]['Predicted'][index] = answer

    # Save predictions
    for i, pred in enumerate(predictions):
        pred.to_csv(pred_dir + "/pred_layer_" + str(i+1) + ".csv", index = False)

if __name__ == "__main__":

    # Train and dev set
    train = "train-v2.0.json"
    dev = "dev-v2.0.json"

    # Usage message
    if len(sys.argv) != 4:
        print('Usage:')
        print('   python3 qa_probes.py [pretrained/fine_tuned] [cpu/gpu] epoches')

    # Model
    if sys.argv[1] == "pretrained":
        model_prefix = "albert-base-v2"
        probe_dir = "pretrained_probes"
        pred_dir = "pretrained_preds"
    elif sys.argv[1] == "fine_tuned":
        model_prefix = "twmkn9/albert-base-v2-squad2"
        probe_dir = "fine_tuned_probes"
        pred_dir = "fine_tuned_preds"

    # Device
    if sys.argv[2] == "cpu":
        device = "cpu"
    elif sys.argv[2] == "gpu":
        device = "cuda"

    # Training epoches
    epoches = int(sys.argv[3])

    # Create probe directory
    if not os.path.exists(probe_dir):
        os.mkdir(probe_dir)

    # Create prediction directory
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    # Set random seed
    torch.manual_seed(1)

    # Train softmax probes
    train_probes(model_prefix,
                 data_dir = "squad-master/data/",
                 filename = dev,
                 probe_dir = probe_dir,
                 epoches = epoches,
                 hidden_dim = 768,
                 max_seq_length = 384,
                 device = device)

    # Generate predictions
    evaluate_probes(model_prefix,
                    data_dir = "squad-master/data/",
                    filename = dev,
                    probe_dir = probe_dir,
                    pred_dir = pred_dir,
                    hidden_dim = 768,
                    max_seq_length = 384,
                    device = device)