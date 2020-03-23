'''
This file is based on the offical evaluation script for SQuAD version 2.0.
'''
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import glob
import csv, json
import pandas as pd

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          # print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def main(data_file, pred_file):
  with open(data_file) as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']
  
  with open(pred_file) as f:
    preds = json.load(f)
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


def convert_preds_to_json(preds_dir):
    for csv_file in glob.glob(preds_dir + "*.csv"):
        prefix = csv_file[:-4]

        data = {}
        with open(csv_file) as f:
            r = csv.DictReader(f)
            for row in r:
                id = row['Id']
                pred = row['Predicted']
                data[id] = pred

        x = json.dumps(data)
        f = open(prefix + ".json", "w")
        f.write(x)
        f.close()

def evaluate(preds_dir, 
             data_path, 
             layers):

    layers_arr = np.arange(layers)+1
    exact, f1 = np.zeros(layers), np.zeros(layers)
    exact_no_ans, f1_no_ans = np.zeros(layers), np.zeros(layers)
    exact_has_ans, f1_has_ans = np.zeros(layers), np.zeros(layers)

    for json_file in glob.glob(preds_dir + "*.json"):

        layer = int(json_file[len(pred_dirs) + len("pred_layer_"):-5])
        l = layer-1 # layer index is layer-1
        exact[l], f1[l], exact_no_ans[l], f1_no_ans[l], exact_has_ans[l], f1_has_ans[l] = main(data_path, json_file)

    results = pd.DataFrame({'layer':layers_arr, 'exact':exact, 'f1':f1, 'exact_no_ans':exact_no_ans, 'f1_no_ans':f1_no_ans, 'exact_has_ans':exact_has_ans, 'f1_has_ans':f1_has_ans})
    print(results)

    save_dir = os.path.abspath(preds_dir+"/../") + "/"
    csv_name = "results.csv"

    results.to_csv(save_dir+csv_name, index = False)

if __name__ == '__main__':

  # Usage message
  if len(sys.argv) != 2:
      print("Usage")
      print("    python3 evaluate.py [model_prefix]")

  # Model prefix
  model_prefix = sys.argv[1]

  model_dir = model_prefix.split("/")[-1]
  
  # Predict using probes for each epoch directory present
  for epoch_dir in sorted(os.listdir(model_dir)):
      for probes_or_preds_dir in sorted(os.listdir(epoch_dir)):
        probes_dir = epoch_dir + "/" + probes_or_preds_dir + "/"
        if os.path.isdir(probes_dir) and probes_dir[-7:] == 'probes/': # confirm it's a probes dir and not preds dir
            
            # Find the associated preds dir
            preds_dir = os.path.abspath(probes_dir+"/../preds/")
            print(preds_dir)
            
            # Convert preds dir csv files to json
            convert_preds_to_json(preds_dir=preds_dir)
            
            # Compare the created json of prediction to the data's truth
            evaluate(preds_dir=preds_dir, 
                     data_path="squad2/dev-v2.0.json", 
                     layers=12)

            print("")
