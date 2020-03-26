import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from probe import Probe

def predict(model_prefix,
            probes_dir,
            preds_dir,
            data_dir,
            data_file,
            layers,
            batch_size,
            hidden_dim,
            max_seq_length,
            device):

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    dev_examples = processor.get_dev_examples(data_dir = data_dir, filename = data_file)

    # Extract dev features
    print("Loading dev features")
    dev_features, dev_dataset = squad_convert_examples_to_features(examples=dev_examples,
                                                                   tokenizer=tokenizer,
                                                                   max_seq_length=max_seq_length,
                                                                   doc_stride=128,
                                                                   max_query_length=64,
                                                                   is_training=False,
                                                                   return_dataset="pt",
                                                                   threads=1)

    # Initialize config and model
    config = AutoConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Load probe for each layer
    print("Loading probes")
    probes = []
    for i in range(layers):
        p = Probe(hidden_dim)
        p.load(probes_dir, i+1, device)
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

    # Initialize predictions
    predictions = []
    for i in range(layers):
        pred = pd.DataFrame()
        pred['Id'] = q_ids
        pred['Predicted'] = [""] * len(dev_examples)
        pred['Question'] = [""] * len(dev_examples)
        pred['Score'] = [0] * len(dev_examples)
        predictions.append(pred)

    # List to keep track of how many unique questions we've seen in each df, questions with
    # contexts longer than max seq len get split into multiple features based on doc_stride
    # a good alternative we may implement later is recording for all features, then simplifying with groupby and max
    # e.g. something like df.sort_values('Score', ascending=False).drop_duplicates(['Question'])
    question_ids = [0]*layers 

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
            
            # Distil does not use token type ids
            if "distil" in model_dir:
                inputs.pop('token_type_ids')

            # ALBERT/BERT/Distilibert forward pass
            idx = batch[3]
            outputs = model(**inputs)
            attention_hidden_states = outputs[2][1:]

            # Compute prediction on eval indices
            for j, index in enumerate(idx):
                index = int(index.item())

                # Extract tokens for the current batch
                tokens = tokenizer.convert_ids_to_tokens(batch[0][j])
                
                # Find where context starts and ends, since we want to predict in context
                context_start = int(max_seq_length - torch.argmax(torch.flip(batch[2][j], [0])).item()) - 1
                context_end = int(torch.argmax(batch[2][j]).item())

                # Find the question, starting right after [CLS] and subtracting 1 to chop off the [SEP] token
                question_start = 1
                question_end = context_start
                question = tokenizer.convert_tokens_to_string(tokens[question_start:question_end-1])

                # For each layer ...
                for i, p in enumerate(probes):

                    # Extract predicted indicies
                    score, start_idx, end_idx = p.predict(attention_hidden_states[i][j].unsqueeze(0), 
                                                          device, 
                                                          threshold=0, 
                                                          context_start=context_start, 
                                                          context_end=context_end)
                    start_idx = int(start_idx[0])
                    end_idx = int(end_idx[0])

                    # Extract predicted answer, converting start tokens to empty strings (no answer)
                    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])
                    if answer == '[CLS]':
                        answer = ''

                    # Check if the question is the same as the last one, if it is go back to the last question id and keep the higher score.
                    # If the question is not already in the dataframe, then assign it to the dataframe.
                    # Note we first handle the case where there are no prior questions by storing since we know there are no duplicates
                    if question_ids[i] == 0:
                        predictions[i].loc[question_ids[i], 'Question'] = question
                        predictions[i].loc[question_ids[i], 'Predicted'] = answer
                        predictions[i].loc[question_ids[i], 'Score'] = score

                    elif (predictions[i].loc[int(question_ids[i]-1), 'Question'] == question):
                        question_ids[i] -= 1  
                        old_score = predictions[i].loc[question_ids[i], 'Score'] 
                        if score > old_score:
                            predictions[i].loc[question_ids[i], 'Predicted'] = answer
                            predictions[i].loc[question_ids[i], 'Score'] = score
                    else:
                        predictions[i].loc[question_ids[i], 'Question'] = question
                        predictions[i].loc[question_ids[i], 'Predicted'] = answer
                        predictions[i].loc[question_ids[i], 'Score'] = score
                    
                    # Increment to new question id (note, for duplicate answers this gets us back to where we were)
                    question_ids[i] += 1

    # Save predictions for each layer
    print("Saving predictions")
    if not os.path.exists(preds_dir):
        os.mkdir(preds_dir)

    for i, pred in enumerate(predictions):
        pred[['Id', 'Predicted']].to_csv(preds_dir + "/layer_" + str(i+1) + ".csv", index = False)

if __name__ == "__main__":

    # Usage message
    if len(sys.argv) != 3:
        print("Usage")
        print("    python3 predict.py [model_prefix] [cpu/gpu]")

    # Model prefix
    model_prefix = sys.argv[1]

    model_dir = model_prefix.split("/")[-1]

    # Distilbert base has 6 layers, while BERT and ALBERT both have 12
    if "distil" in model_dir:
        layers = 6
    else:
        layers = 12

    # Device
    device = sys.argv[2]

    if device == "cpu":
        device = "cpu"
    elif device == "gpu":
        device = "cuda"

    # Predict using probes for each epoch directory present
    for epoch_dir in sorted(os.listdir(model_dir)):
        full_epoch_dir = model_dir + "/" + epoch_dir # full path to epoch dir
        for probes_or_preds_dir in sorted(os.listdir(full_epoch_dir)):
            full_probes_or_preds_dir = model_dir + "/" + epoch_dir + "/" + probes_or_preds_dir # full path to probes or preds dir
            if os.path.isdir(full_probes_or_preds_dir) and full_probes_or_preds_dir[-6:] == 'probes': # confirm it's a probes dir
                probes_dir = full_probes_or_preds_dir
                preds_dir = os.path.abspath(probes_dir+"/../preds")
                predict(model_prefix = model_prefix,
                        probes_dir = probes_dir,
                        preds_dir = preds_dir,
                        data_dir = "squad2/",
                        data_file = "dev-v2.0.json",
                        layers = layers,
                        batch_size = 8,
                        hidden_dim = 768,
                        max_seq_length = 384,
                        device = device)
                print("")
