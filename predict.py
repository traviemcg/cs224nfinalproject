import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
from transformers import *
from transformers.data.processors.squad import SquadV2Processor
from probe import Probe

def eval_model(model_prefix,
               probe_dir,
               pred_dir,
               data_dir,
               dev_file,
               layers,
               hidden_dim,
               batch_size,
               max_seq_length,
               device):

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    dev_examples = processor.get_dev_examples(data_dir = data_dir, filename = dev_file)

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

    # Load probe
    print("Loading probes")
    probes = []
    for i in range(layers):
        p = Probe(hidden_dim)
        p.load(probe_dir, i+1, device)
        probes.append(p)

    # Extract IDs
    print("Extracting dev IDs")
    n = len(dev_examples)
    q_ids = []
    for i in range(n):
        q_ids.append(dev_examples[i].qas_id)

    # Initialize dev data loader
    eval_sampler = SequentialSampler(dev_dataset)
    eval_dataloader = DataLoader(dev_dataset, sampler = eval_sampler, batch_size = batch_size)

    print(len(dev_examples))
    print(len(dev_dataset)) # this is 6398, we were expecting 6078 ?
    print(len(eval_dataloader)) # this is 1600, which implies 1600*4=6400 examples. But we only have 6078...

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

            # Compute prediction on eval indices
            for j, index in enumerate(idx):
                index = int(index.item())

                if index >= n:
                    break
                
                if index == 6077:
                    print(index, dev_examples[index].qas_id)
                    tokens = tokenizer.convert_ids_to_tokens(batch[0][j])
                    print(tokenizer.convert_tokens_to_string(tokens))

                for i, p in enumerate(probes):

                    # Find where context starts and ends, since we want to predict in context
                    context_start = int(max_seq_length - torch.argmax(torch.flip(batch[2][j], [0])).item())
                    context_end = int(torch.argmax(batch[2][j]).item())

                    # Extract predicted indicies
                    start_idx, end_idx = p.predict(attention_hidden_states[i][j].unsqueeze(0), device, threshold=0, context_start=context_start, context_end=context_end)
                    start_idx = int(start_idx[0])
                    end_idx = int(end_idx[0])

                    # Extract predicted answer
                    tokens = tokenizer.convert_ids_to_tokens(batch[0][j])
                    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

                    # No answer
                    if answer == '[CLS]':
                        answer = ''

                    # Populate output
                    predictions[i]['Predicted'][predictions[i]['Id']==dev_examples[index].qas_id] = answer

    # Save predictions
    print("Saving predictions")
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for i, pred in enumerate(predictions):
        pred.to_csv(pred_dir + "/pred_layer_" + str(i+1) + ".csv", index = False)

if __name__ == "__main__":

    # Usage message
    if len(sys.argv) != 4:
        print("Usage")
        print("    python3 predict.py [exper/probes] [experiment/probe dir] [cpu/gpu]")

    # Whether passing preds or exper dir
    use_probes_or_exper_dir = sys.argv[1]

    # Directory to use for preds or exper
    experiment_dir = sys.argv[2]
    if experiment_dir[-1] != "/":
        experiment_dir = experiment_dir + "/"

    # Device
    if sys.argv[3] == "cpu":
        device = "cpu"
    elif sys.argv[3] == "gpu":
        device = "cuda"

    # Do evaluation for whole experiment
    if use_probes_or_exper_dir == "exper":
        epoch_names = sorted(os.listdir(experiment_dir))
        for epoch_name in epoch_names:
            if "pretrained" in epoch_name:
                pretrained_or_fine_tuned = "pretrained"
                model_prefix = "albert-base-v2"
            if "fine_tuned" in epoch_name:
                pretrained_or_fine_tuned = "fine_tuned"
                model_prefix = "twmkn9/albert-base-v2-squad2"

            epoch_dir = experiment_dir + epoch_name
            if os.path.isdir(epoch_dir):
                for possible_probe_name in os.listdir(epoch_dir):
                    probe_dir = epoch_dir + "/" + possible_probe_name + "/"
                    if os.path.isdir(probe_dir) and probe_dir[-7:] == 'probes/':
                        print(probe_dir)
                        pred_dir = os.path.abspath(probe_dir+"/../" + pretrained_or_fine_tuned + "_preds/")
                        eval_model(model_prefix,
                                   probe_dir,
                                   pred_dir,
                                   data_dir = "squad-master/data/",
                                   dev_file = "dev-v2.0.json",
                                   layers = 12,
                                   hidden_dim = 768,
                                   batch_size = 4,
                                   max_seq_length = 384,
                                   device = device)
                        print("")
  
    # Do prediction for single probes directory
    elif use_probes_or_exper_dir == "probes":
        probe_dir = experiment_dir
        if "pretrained" in probe_dir:
            model_prefix = "albert-base-v2"
            pretrained_or_fine_tuned = "pretrained"
        if "fine_tuned" in probe_dir:
            model_prefix = "twmkn9/albert-base-v2-squad2"
            pretrained_or_fine_tuned = "fine_tuned"
        pred_dir = os.path.abspath(probe_dir+"/../"+pretrained_or_fine_tuned+"_preds/")

        eval_model(model_prefix,
                   probe_dir,
                   pred_dir,
                   data_dir = "squad-master/data/",
                   dev_file = "dev-v2.0.json",
                   layers = 12,
                   hidden_dim = 768,
                   batch_size = 4,
                   max_seq_length = 384,
                   device = device)
