import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import *
from transformers.data.processors.squad import SquadV2Processor
from probe import Probe

def send_epochs(model_prefix,
                 data_dir,
                 train_file,
                 epoch_dir,
                 probe_dir,
                 pred_dir,
                 epochs,
                 batch_size = 8,
                 layers = 12,
                 hidden_dim = 768,
                 max_seq_length = 384,
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

    # Load class weight file, if it exists
    if os.path.isfile('class_weights.pkl'):
        weight = torch.load('class_weights.pkl', map_location=device).to(device)
    else:
        weight = None

    # Initialize ALBERT model
    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
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

        # Make epoch directory
        it_epoch_dir = epoch_dir + "_" + str(epoch + 1)
        if not os.path.exists(it_epoch_dir):
            os.mkdir(it_epoch_dir)

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
            attention_hidden_states = outputs[3][1:]

            # Initialize batch loss for each probe to zero
            batch_loss = torch.ones(layers, dtype=torch.float32, device=device, requires_grad=True)

            # Update probes
            for j in range(batch[7].shape[0]):
                start = batch[3][j].clone().unsqueeze(0).to(device)
                end  = batch[4][j].clone().unsqueeze(0).to(device)

                # Get loss for each example in batch
                for i, p in enumerate(probes):
                    batch_loss[i] = batch_loss[i] + p.train(attention_hidden_states[i][j].unsqueeze(0), start, end, device, weight=weight)

            # Take gradient steps for batch
            for i, p in enumerate(probes):
                p.step(batch_loss[i], device)

        # Save probes after each epoch
        print("Epoch complete, saving probes")
        it_probe_dir = it_epoch_dir + "/" + probe_dir
        if not os.path.exists(it_probe_dir):
            os.mkdir(it_probe_dir)
        for i, p in enumerate(probes):
            p.save(it_probe_dir, i + 1)

if __name__ == "__main__":

    # Train set
    train = "train-v2.0.json"

    # Usage message
    if len(sys.argv) != 4:
        print('Usage:')
        print('   python3 train.py [pretrained/fine_tuned/both] [cpu/gpu] epochs')

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
    elif sys.argv[1] == "both":
        model_prefixes = ["albert-base-v2", "twmkn9/albert-base-v2-squad2"]
        epoch_dirs = ["pretrained_epoch", "fine_tuned_epoch"]
        probe_dirs = ["pretrained_probes", "fine_tuned_probes"]
        pred_dirs = ["pretrained_preds", "fine_tuned_preds"]

    # Device
    if sys.argv[2] == "cpu":
        device = "cpu"
    elif sys.argv[2] == "gpu":
        device = "cuda"

    # Training epochs
    epochs = int(sys.argv[3])

    # Set random seed
    torch.manual_seed(1)

    # Send epochs
    for i in range(len(model_prefixes)):
        model_prefix = model_prefixes[i]
        epoch_dir = epoch_dirs[i]
        probe_dir = probe_dirs[i]
        pred_dir = pred_dirs[i]
        send_epochs(model_prefix,
                    data_dir = "squad-master/data/",
                    train_file = train,
                    epoch_dir = epoch_dir,
                    probe_dir = probe_dir,
                    pred_dir = pred_dir,
                    epochs = epochs,
                    hidden_dim = 768,
                    max_seq_length = 384,
                    device = device)