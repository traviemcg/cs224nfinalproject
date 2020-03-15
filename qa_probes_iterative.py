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

def send_epochs(model_prefix,
                 data_dir,
                 train_file,
                 dev_file,
                 epoch_dir,
                 probe_dir,
                 pred_dir,
                 epochs,
                 batch_size = 4,
                 layers = 12,
                 hidden_dim = 768,
                 max_seq_length = 384,
                 device = 'cpu'):


    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    train_examples = processor.get_train_examples(data_dir = data_dir, filename = dev_file) # change to train file
    dev_examples = processor.get_train_examples(data_dir = data_dir, filename = dev_file)
    
    train_examples = train_examples[0:8]
    dev_examples = dev_examples[0:8]

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

    # Extract dev features
    print("Loading dev features")
    dev_features, dev_dataset = squad_convert_examples_to_features(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
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
        p = MultiSoftmaxRegression(hidden_dim, len(train_examples), epochs)
        probes.append(p)

    # Extract IDs
    print("Extracting dev IDs")
    n = len(dev_examples)
    q_ids = []
    for i in range(n):
        q_ids.append(dev_examples[i].qas_id)

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

        # Save probes after each epoch
        print("Epoch complete, saving probes")
        it_probe_dir = it_epoch_dir + "/" + probe_dir
        if not os.path.exists(it_probe_dir):
            os.mkdir(it_probe_dir)
        for i, p in enumerate(probes):
            p.save(it_probe_dir, i + 1)

        # Initialize dev data loader
        eval_sampler = SequentialSampler(dev_dataset)
        eval_dataloader = DataLoader(dev_dataset, sampler = eval_sampler, batch_size = batch_size)

        # Initialize predictions
        predictions = []
        for i in range(layers):
            pred = pd.DataFrame()
            pred['Id'] = q_ids
            pred['Predicted'] = [""] * len(dev_examples)
            predictions.append(pred)

        # Evaluation batches
        print("Predicting on dev set")
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

                        # Throw out invalid predictions
                        tokens = tokenizer.convert_ids_to_tokens(batch[0][j])
                        context_len = batch[2][j].sum()
                        question_len = max_seq_length - context_len
                        if start_idx >= len(tokens):
                            start_idx, end_idx = 0, 0
                        if end_idx >= len(tokens):
                            start_idx, end_idx = 0, 0
                        if end_idx < start_idx:
                            start_idx, end_idx = 0, 0
                        max_answer_length = 22
                        length = end_idx - start_idx + 1
                        if length > max_answer_length:
                            start_idx, end_idx = 0, 0

                        # Extract predicted answer
                        answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

                        # No answer
                        if answer == '[CLS]':
                            answer = ''

                        # Populate output
                        predictions[i]['Predicted'][index] = answer

        # Save predictions
        print("Saving predictions")
        it_pred_dir = it_epoch_dir + "/" + pred_dir
        if not os.path.exists(it_pred_dir):
            os.mkdir(it_pred_dir)
        for i, pred in enumerate(predictions):
            pred.to_csv(it_pred_dir + "/pred_layer_" + str(i+1) + ".csv", index = False)

if __name__ == "__main__":

    # Train and dev set
    train = "train-v2.0.json"
    dev = "dev-v2.0.json"

    # Usage message
    if len(sys.argv) != 4:
        print('Usage:')
        print('   python3 qa_probes_iterative.py [pretrained/fine_tuned] [cpu/gpu] epochs')

    # Model
    if sys.argv[1] == "pretrained":
        model_prefix = "albert-base-v2"
        epoch_dir = "pretrained_epoch"
        probe_dir = "pretrained_probes"
        pred_dir = "pretrained_preds"
    elif sys.argv[1] == "fine_tuned":
        model_prefix = "twmkn9/albert-base-v2-squad2"
        epoch_dir = "fine_tuned_epoch"
        probe_dir = "fine_tuned_probes"
        pred_dir = "fine_tuned_preds"

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
    send_epochs(model_prefix,
                 data_dir = "squad-master/data/",
                 train_file = train,
                 dev_file = dev,
                 epoch_dir = epoch_dir,
                 probe_dir = probe_dir,
                 pred_dir = pred_dir,
                 epochs = epochs,
                 hidden_dim = 768,
                 max_seq_length = 384,
                 device = device)