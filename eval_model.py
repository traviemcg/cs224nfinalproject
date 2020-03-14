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

def eval_model(model_prefix,
               probe_dir,
               pred_dir,
               epochs,
               data_dir = "squad-master/data/",
               dev_file = "dev-v2.0.json",
               layers = 12,
               train_size = 130139,
               hidden_dim = 768,
               batch_size = 4,
               max_seq_length = 384,
               device = 'gpu'):

    # Load probe
    print("Loading probes")
    probes = []
    for i in range(layers):
        probe = MultiSoftmaxRegression(768, 130139, 5)
        probe.load(probe_dir, i)

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    dev_examples = processor.get_train_examples(data_dir = data_dir, filename = dev_file)

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

    # Extract IDs
    print("Extracting dev IDs")
    n = len(dev_examples)
    q_ids = []
    for i in range(n):
        q_ids.append(dev_examples[i].qas_id)

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

                    # Extract predicted answer
                    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

                    # No answer
                    if answer == '[CLS]':
                        answer = ''

                    # Populate output
                    predictions[i]['Predicted'][index] = answer

    # Save predictions
    print("Saving predictions")
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for i, pred in enumerate(predictions):
        pred.to_csv(pred_dir + "/pred_layer_" + str(i+1) + ".csv", index = False)

if __name__ == "__main__":

    dev = "dev-v2.0.json"

    if len(sys.argv) != 4:
        print("Usage")
        print("    [pretrained/fine_tuned]probe_dir pred_dir epochs")

    if sys.argv[1] == "pretrained":
        model_prefix = "albert-base-v2"
    elif sys.argv[1] == "fine_tuned":
        model_prefix = "twmkn9/albert-base-v2-squad2"

    probe_dir = sys.argv[2]
    pred_dir = sys.argv[3]
    epochs = int(sys.argv[4])

    eval_model(model_prefix, probe_dir, pred_dir, epochs)