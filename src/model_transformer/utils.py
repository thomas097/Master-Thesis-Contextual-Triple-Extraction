import glob
import json
import re
import random
from collections import defaultdict
import numpy as np


## IO

def load_annotations(path):
    annotations = []
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            annotations.append(json.load(file))
    return annotations


## Argument Extraction

def triple_to_bio_tags(annotation, arg):
    """ Converts the index-based annotation scheme to a vector of BIO tags
        for some argument (arg=0->subj, arg=1->pred, arg=2->obj).
    """
    turns = annotation['tokens']
    triples = annotation['annotations']
    num_tokens = sum([len(turn) + 1 for turn in turns])  #+1 for <eos>

    # Create label vector containing a value for each token in dialogue
    mask = np.zeros(num_tokens, dtype=np.uint8)

    # Label annotated arguments as BIO tags
    for triple in triples:
        for j, (turn_id, token_id) in enumerate(triple[arg]):
            k = sum([len(t) + 1 for t in turns[:turn_id]]) + token_id # index k of token in dialog
            mask[k] = 1 if j == 0 else 2
    return mask


def bio_tags_to_tokens(tokens, mask, one_hot=False):
    """ Converts a vector of BIO-tags into spans of tokens.
    """
    out = []
    span = []
    for i, token in enumerate(tokens):
        pred = mask[i]

        # Reverse one-hot encoding (optional)
        if one_hot:
            pred = np.argmax(pred)

        if pred == 1: # Beginning
            span = re.sub('[^\w\d\- ]+', '', ' '.join(span))
            out.append(span)
            span = [token]

        elif pred == 2: # Inside
            span.append(token)

    if span:
        span = re.sub('[^\w\d\- ]+', '', ' '.join(span))
        out.append(span)

    # Remove empty strings and duplicates
    return set([span for span in out if span.strip()])


## Triple Scoring

def extract_triples(annotation):
    turns = annotation['tokens']
    triple_ids = annotation['annotations']

    arguments = defaultdict(list)
    triples = []
    labels = []

    triple_ids = [t[:4] for t in triple_ids]

    for subj, pred, obj, neg in triple_ids:
        # Extract tokens belonging to triple arguments
        subj = ' '.join(turns[i][j] for i, j in subj) if subj else ''
        pred = ' '.join(turns[i][j] for i, j in pred) if pred else ''
        obj = ' '.join(turns[i][j] for i, j in obj) if obj else ''

        if subj or pred or obj:

            if not neg:
                triples += [(subj, pred, obj)]
                labels += [1]
            else:
                triples += [(subj, pred, obj)] * 6 # oversampling
                labels += [2] * 6

            arguments['subjs'].append(subj)
            arguments['preds'].append(pred)
            arguments['objs'].append(obj)

    # Create negative examples (i.e. not entailed)
    n = int(len(triples) * 0.6)
    for i in range(100):
        s = random.choice(arguments['subjs'])
        p = random.choice(arguments['preds'])
        o = random.choice(arguments['objs'])

        # Check if the triple was already generated
        if (s, p, o) not in triples and s and p and o:
            triples += [(s, p, o)] # not entailed
            labels += [0]
            n -= 1

        # Create as many on-entailed examples as entailed ones
        if n == 0:
            break

    return triples, labels
