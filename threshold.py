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
import argparse
import collections
import json
import numpy as np
import re
import string
import sys
import glob
import csv, json
from random import sample
from evaluate import *

def main_alt(dataset, preds):
    '''
    @param dataset
    @param preds, dictionary with key: question id, value: text prediction

    @return metrics for subset of question ids predicted on
    '''
    # with open(data_file) as f:
    #     dataset_json = json.load(f)
    #     dataset = dataset_json['data']
    na_probs = {k: 0.0 for k in preds}

    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    has_ans_qids_in_pred = [qid for qid in has_ans_qids if qid in preds]
    no_ans_qids_in_pred = [qid for qid in no_ans_qids if qid in preds]

    exact_raw, f1_raw = get_raw_scores(dataset, preds)

    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, 1.0)

    out_eval = make_eval_dict(exact_thresh, f1_thresh)

    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids_in_pred)
        merge_eval(out_eval, has_ans_eval, 'has_ans')

    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids_in_pred)
        merge_eval(out_eval, no_ans_eval, 'no_ans')

    exact, f1 = out_eval['exact'], out_eval['f1']
    exact_no_ans, f1_no_ans = out_eval['no_ans_exact'], out_eval['no_ans_f1']
    exact_has_ans, f1_has_ans = out_eval['has_ans_exact'], out_eval['has_ans_f1']

    return exact, f1, exact_no_ans, f1_no_ans, exact_has_ans, f1_has_ans


def eval_thresholds(probe_dir,
                    model_prefix,
                    thresholds = [-1, 0, 1],
                    full_set = False,
                    n = 100,
                    trials = 2,
                    max_seq_length = 384,
                    hidden_dim = 768,
                    layers = 12,
                    batch_size = 4,
                    data_dir = "squad-master/data/",
                    train_file = "dev-v2.0.json",
                    device = "cpu"):

    # Extract examples
    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_dev_examples(data_dir = data_dir, 
                                          filename = train_file)

    # Initialize (AL)BERT model
    if "albert" in model_prefix:
        config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    else:
        config = BertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)
    model = torch.nn.DataParallel(model)

    # Load probes
    print("Loading probes")
    probes = []
    for i in range(layers):
        p = Probe(hidden_dim)
        p.load(probe_dir, i+1, device)
        probes.append(p)

    # Initialize results
    result = {}
    for t in thresholds:
        result[t] = np.zeros((layers, 6))

    # First argument for main_alt
    with open(data_dir + train_file) as f:
        dataset_json = json.load(f)
        main_alt_dataset = dataset_json['data']

    # If full_set specified, evaluate 1 trial on full set
    if full_set:
        trials = 1
        it_examples = examples

    # Execute trials
    for it in range(trials):

        # Randomly sample n examples
        if not full_set:
            it_examples = sample(examples, n)

        # Extract features
        features, dataset = squad_convert_examples_to_features(examples=it_examples,
                                                               tokenizer=tokenizer,
                                                               max_seq_length=max_seq_length,
                                                               doc_stride=128,
                                                               max_query_length=64,
                                                               is_training=False,
                                                               return_dataset="pt",
                                                               threads=1)

        # Extract question IDs
        q_ids = []
        for i in range(n):
            q_ids.append(it_examples[i].qas_id)

        # Initialize data loader
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, 
                                     sampler = eval_sampler, 
                                     batch_size = batch_size)

        # Initialize predictions
        predictions_dict = {}
        for t in thresholds:
            predictions = []
            for i in range(layers):
                pred = pd.DataFrame()
                pred['Id'] = q_ids
                pred['Predicted'] = [""] * len(it_examples)
                pred['Question'] = [""] * len(it_examples)
                pred['Score'] = [0] * len(it_examples)
                predictions.append(pred)
            predictions_dict[t] = predictions

        # List to keep track of how many unique questions we've seen in each df, questions with
        # contexts longer than max seq len get split into multiple features based on doc_stride
        # a good alternative we may implement later is recording for all features, then simplifying with groupby and max
        # e.g. something like df.sort_values('Score', ascending=False).drop_duplicates(['Question'])
        question_ids = {}
        for t in thresholds:
            question_ids[t] = [0]*layers 

        print("Trial: {}".format(it))
        for batch in tqdm(eval_dataloader, desc = "Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
        
            with torch.no_grad():
                inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }

                # (AL)BERT forward pass
                idx = batch[3]
                outputs = model(**inputs)
                attention_hidden_states = outputs[2][1:]

                # Compute predictions on eval indicies
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

                    # Iterate over probes
                    for i, p in enumerate(probes):

                        # Iterate over thresholds
                        for t in thresholds:
                                                
                            # Extract predicted indicies
                            score, start_idx, end_idx = p.predict(attention_hidden_states[i][j].unsqueeze(0), 
                                                                  device, 
                                                                  threshold=t, 
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
                            if question_ids[t][i] == 0:
                                predictions_dict[t][i].loc[question_ids[t][i], 'Question'] = question
                                predictions_dict[t][i].loc[question_ids[t][i], 'Predicted'] = answer
                                predictions_dict[t][i].loc[question_ids[t][i], 'Score'] = score

                            elif (predictions_dict[t][i].loc[int(question_ids[t][i]-1), 'Question'] == question):
                                question_ids[t][i] -= 1  
                                old_score = predictions_dict[t][i].loc[question_ids[t][i], 'Score'] 
                                if score > old_score:
                                    predictions_dict[t][i].loc[question_ids[t][i], 'Predicted'] = answer
                                    predictions_dict[t][i].loc[question_ids[t][i], 'Score'] = score
                            else:
                                predictions_dict[t][i].loc[question_ids[t][i], 'Question'] = question
                                predictions_dict[t][i].loc[question_ids[t][i], 'Predicted'] = answer
                                predictions_dict[t][i].loc[question_ids[t][i], 'Score'] = score
                            
                            # Increment to new question id (note, for duplicate answers this gets us back to where we were)
                            question_ids[t][i] += 1

        # Evaluate predictions
        for t in thresholds:
            for i in range(layers):
                preds = dict(zip(predictions_dict[t][i].Id, predictions_dict[t][i].Predicted))
                result[t][i] += main_alt(main_alt_dataset, preds)

    # Normalize results
    for t in thresholds:
        result[t] /= trials

    return result
