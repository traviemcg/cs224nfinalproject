import numpy as np
from transformers import *
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadV2Processor
import sys
from SoftmaxRegression import MultiSoftmaxRegression



def train_probes(model_prefix,
                 data_dir,
                 filename,
                 epoches = 1,
                 hidden_dim = 768,
                 max_seq_length = 384):
    '''
       Trains softmax probe corresponding to each layer of Albert

    '''

    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir = data_dir, filename = filename)

    examples = examples[:2]

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Initialize probes
    probe_1 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_2 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_3 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_4 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_5 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_6 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_7 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_8 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_9 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_10 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_11 = MultiSoftmaxRegression(max_seq_length, hidden_dim)
    probe_12 = MultiSoftmaxRegression(max_seq_length, hidden_dim)

    for epoch in range(epoches):

        print("TRAINING EPOCH: {}".format(epoch))

        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=1)

        for batch in tqdm(train_dataloader, desc = "Iteration"):
            model.eval()
            batch = tuple(t.to('cpu') for t in batch)
            
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
                attention_hidden_states = outputs[2][1:]

                # Extract labels
                print(batch[5])


                # Update probes

    # Save NN's


def evaluate_probes(model_prefix,
                    data_dir,
                    filename):


    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir = data_dir, filename = filename)

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

    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = 4)

    # multi-gpu evaluate
    model = torch.nn.DataParallel(model)

    # Load probes

    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)
        
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

            # Generate predictions 

            # Write predictions to file
            # for i, j in enumerate(idx):
            #     f.write("{}, {}".format(j, pred[i])


if __name__ == "__main__":

    # Train and dev set
    train = "train-v2.0.json"
    dev = "dev-v2.0.json"

    # Model
    if sys.argv[1] == "albert-base-v2":
        model_prefix = "albert-base-v2"
        output_prefix = "base-v2_" + sys.argv[1]

    # Train softmax probe
    train_probes(model_prefix,
                 data_dir = "squad-master/data/",
                 filename = train,
                 epoches = 1,
                 hidden_dim = 768,
                 max_seq_length = 384)