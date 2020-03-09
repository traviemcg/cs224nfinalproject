import numpy as np
from transformers import *
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadV2Processor
import sys


def extract_layers(model_prefix, 
                   data_dir, 
                   filename,
                   output_prefix,
                   layers = 12,
                   hidden_dim = 4096):
    '''

    @param model_prefix, Model prefix (ahotrod/albert_xxlargev1_squad2_512)
    @param data_dir, directory with data
    @param filename, file to be converted

    '''

    tokenizer = AutoTokenizer.from_pretrained(model_prefix)
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir = data_dir, filename = filename)

    examples = examples[:5]

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

    # Initialize result
    # result = []
    # for i in range(layers):
    #     result.append(np.zeros((len(examples), 384, hidden_dim)))

    config = AlbertConfig.from_pretrained(model_prefix, output_hidden_states = True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_prefix, config = config)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = 1)

    l = output_prefix + "_layer_"

    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)
        
        with torch.no_grad():
            inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

            idx = batch[3].numpy()[0]
            outputs = model(**inputs)
            attention_hidden_states = outputs[2][1:]

            # Populate output
            for i in range(layers):
                h = attention_hidden_states[i].numpy()[0]
                f = open(l + str(i+1), 'a')
                f.write("{}, {}\n".format(idx, h.tolist()))
                f.close()


                # with open(layer_file + str(i + 1), 'w') as f:
                #     h = attention_hidden_states[i].numpy()[0]
                #     f.write("{}, {}\n".format(idx, h.tolist()))

            
    # Save outputs 
    # for i in range(layers):
    #     np.save(output_prefix + "_layer_" + str(i + 1), result[i])

if __name__ == "__main__":

    # Train or dev set
    if sys.argv[1] == "train":
        filename = "train-v2.0.json"
    elif sys.argv[1] == "dev":
        filename = "dev-v2.0.json"

    # Model
    if sys.argv[2] == "xxlargev1_squad2_512":
        model_prefix = "ahotrod/albert_xxlargev1_squad2_512"
        output_prefix = "xxlargev1_squad2_512_" + sys.argv[1]
        hidden_dim = 4096
    elif sys.argv[2] == "albert-xxlarge-v1":
        model_prefix = "albert-xxlarge-v1"
        output_prefix = "xxlarge-v1_" + sys.argv[1]
        hidden_dim = 4096
    elif sys.argv[2] == "albert-base-v2":
        model_prefix = "albert-base-v2"
        output_prefix = "base-v2_" + sys.argv[1]
        hidden_dim = 768

    extract_layers(model_prefix = model_prefix,
                   data_dir = "squad-master/data/",
                   filename = filename,
                   output_prefix = output_prefix,
                   hidden_dim = hidden_dim)