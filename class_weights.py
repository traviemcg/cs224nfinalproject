import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import *
from transformers.data.processors.squad import SquadV2Processor

def get_weights(model_prefix,
                data_dir,
                train_file,
                max_seq_length,
                device = 'cpu'):


    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    train_examples = processor.get_train_examples(data_dir = data_dir, filename = train_file)

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

    # Initialize train data loader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler)

    counts = torch.zeros([2, max_seq_length], dtype=torch.int32, device=device)

    # Training batches
    for batch in tqdm(train_dataloader, desc = "Iteration"):
        batch = tuple(t.to(device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        # Count occurences for each start and end index
        counts[0, batch[3]] += 1
        counts[1, batch[4]] += 1
    
    # Create a weight matrix as (# samples total)/(# samples index) for both start and stop
    class_weights = (torch.zeros([2, max_seq_length], dtype=torch.float32, device=device)+1.0)
    class_weights = class_weights*torch.sum(counts[0, :])
    class_weights = class_weights/counts
    
    print("counts", counts, "\n")
    print("class weights", class_weights, "\n")
    torch.save(class_weights, 'class_weights.pkl')

if __name__ == "__main__":

    # Train set
    train = "train-v2.0.json"

    # Usage message
    if len(sys.argv) != 3:
        print('Usage:')
        print('   python3 class_weights.py [pretrained/fine_tuned] [cpu/gpu]')

        # Model
    if sys.argv[1] == "pretrained":
        model_prefixes = ["albert-base-v2"]
        epoch_dirs = ["pretrained_epoch"]
        probe_dirs = ["pretrained_probes"]
        pred_dirs = ["pretrained_preds"]
    elif sys.argv[1] == "fine_tuned":
        model_prefixes = ["twmkn9/albert-base-v2-squad2"]
        epoch_dirs = ["fine_tuned_epoch"]
        probe_dirs = ["fine_tuned_probes"]
        pred_dirs = ["fine_tuned_preds"]

    # Device
    if sys.argv[2] == "cpu":
        device = "cpu"
    elif sys.argv[2] == "gpu":
        device = "cuda"

    # Calculate class weights
    for i in range(len(model_prefixes)):
        model_prefix = model_prefixes[i]
        epoch_dir = epoch_dirs[i]
        probe_dir = probe_dirs[i]
        pred_dir = pred_dirs[i]
        get_weights(model_prefix,
                    data_dir = "squad-master/data/",
                    train_file = train,
                    max_seq_length = 384,
                    device = device)