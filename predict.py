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

def squad_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    print("1: {}".format(len(features)))
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    print("2: {}".format(len(features)))
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                        "is_impossible": ex.is_impossible,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {
                    "start_position": tf.int64,
                    "end_position": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            ),
        )

    return features

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

    
    print(len(dev_examples)) # this is 6078, so sometime between transfer from this to dev dataset it gets bigger
    print(len(dev_features)) # this is also 6398, so at least dev features aligns with dev dataset
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
