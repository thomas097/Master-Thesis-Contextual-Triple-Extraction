import json
import glob
from pathlib import Path
import numpy as np


def load_annotations(path):
    annotations = dict()
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not data['skipped']:
                filename = Path(fname).parts[-1]
                annotations[filename] = data
    return annotations


def token_argument_assignments(annotation, arg='subj'):
    # Flatten dialogue and assign O (outside) labels to each
    tokens = [t for ts in annotation['tokens'] for t in ts]
    labels = [0 for _ in tokens]

    # Determine turn lengths
    turn_lens = np.cumsum([0] + [len(t) for t in annotation['tokens']])

    # Select triple index according to arg
    indices = {'subj': 0, 'pred': 1, 'obj': 2, 'pol': 3, 'cert': 4}
    idx = indices[arg]

    # Populate sequence
    for triple in annotation['annotations']:
        for i, j in triple[idx]:
            # Compute index in flattened sequence
            k = turn_lens[i] + j
            labels[k] = 1
    return labels


def triples(annotation, corrected=False):
    tokens = annotation['tokens']
    triples = []
    for subj, pred, obj, pol, cert in annotation['annotations']:
        subj = [tokens[i][j] for i, j in subj]
        pred = [tokens[i][j] for i, j in pred]
        obj = [tokens[i][j] for i, j in obj]

        if corrected:
            subj = correct_auxs_and_particles(subj)
            pred = correct_auxs_and_particles(pred)
            obj = correct_auxs_and_particles(obj)

        if subj or pred or obj:
            triples.append((subj, pred, obj))
    return triples


def arguments(annotation, arg, corrected=False):
    indices = {'subj': 0, 'pred': 1, 'obj': 2, 'pol': 3, 'cert': 4}
    tokens = annotation['tokens']
    args = []
    for triple in annotation['annotations']:
        arg_tokens = [tokens[x][y] for x, y in triple[indices[arg]]]

        if corrected:
            arg_tokens = correct_auxs_and_particles(arg_tokens)

        if arg_tokens:  # check whether they are empty
            args.append(arg_tokens)
    return args


def correct_auxs_and_particles(tokens):
    # Remove stopwords from the beginning of the argument (often wrong by annotators)
    stopwords = ["'v", 'have', "'m", "am", 'was', 'is', 'has', 'to', 'from', "'s", "is", "that",
                 'often', "'d"]
    if len(tokens) > 1 and tokens[0] in stopwords:
        tokens = tokens[1:]
    return tokens