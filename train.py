import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import *
from transformers.data.processors.squad import SquadV2Processor
from probe import Probe

def train(model_prefix,
          model_dir,
          data_dir,
          data_file,
          epochs,
          layers,
          batch_size,
          hidden_dim,
          max_seq_length,
          device):

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    train_examples = processor.get_train_examples(data_dir=data_dir, filename=data_file)

    # Extract train features
    print("Loading train features")
    train_features, train_dataset = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset="pt",
        threads=1,
    )

    # Initialize ALBERT/BERT model
    if "albert" in model_prefix:
        config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    elif "bert" in model_prefix:
        config = BertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Initialize probes
    print("Initializing probes")
    probes = []
    for i in range(layers):
        p = Probe(hidden_dim)
        probes.append(p)

    # Training epochs
    for epoch in range(epochs):

        print("Training epoch: {}".format(epoch + 1))

        # Initialize train data loader
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        # Training batches
        for batch in tqdm(train_dataloader, desc = "Iteration"):

            # Get batch on the right device and prepare input dict
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            
            # Albert forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract hiddent states
            all_layer_hidden_states = outputs[3][1:] # (layers, batch_size, max_seq_len, hidden_size)

            # Get labels, and update probes for batch
            start = batch[3] # (batch_size)
            end  = batch[4] # (batch_size)

            for i, p in enumerate(probes):
                hiddens = all_layer_hidden_states[i] # (batch_size, max_seq_len, hidden_size)
                p.train(hiddens, start, end, device, weight=weight)

        # Save probes after each epoch
        print("Epoch complete, saving probes")
        epoch_dir = model_dir + "/epoch_" + str(epoch + 1)
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        
        probes_dir = epoch_dir + "/probes"
        if not os.path.exists(probes_dir):
            os.mkdir(probes_dir)
        
        # Save probes for each layer, both start and end index
        for i, p in enumerate(probes):
            p.save(probes_dir, i + 1)

if __name__ == "__main__":

    # Usage message
    if len(sys.argv) != 4:
        print('Usage:')
        print('   python3 train.py [model_prefix] [cpu/gpu] epochs')

    # Model prefix
    model_prefix = sys.argv[1]

    model_dir = model_prefix.split("/")[-1]
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Device
    device = sys.argv[2]

    if device == "cpu":
        device = "cpu"
    elif device == "gpu":
        device = "cuda"

    # Training epochs
    epochs = int(sys.argv[3])

    # Set random seed
    torch.manual_seed(1)

    # Send epochs
    train(model_prefix = model_prefix,
          model_dir = model_dir,
          data_dir = "squad2/",
          data_file = "train-v2.0.json",
          epochs = epochs,
          layers = 12,
          batch_size = 8,
          hidden_dim = 768,
          max_seq_length = 384,
          device = device)