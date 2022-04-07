import glob
import json
import re
import random
import numpy as np

from collections import defaultdict
from copy import deepcopy

## IO

def load_annotations(path):
    annotations = []
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not data['skipped']:
                data['tokens'] = [[t for t in seq if t != '[unk]'] for seq in data['tokens']]  # remove [unk]
                annotations.append(data)
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
            span = re.sub('[^\w\d\-]+', ' ', ''.join(span)).strip()
            out.append(span)
            span = [token]

        elif pred == 2: # Inside
            span.append(token)

    if span:
        span = re.sub('[^\w\d\-]+', ' ', ''.join(span)).strip()
        out.append(span)

    # Remove empty strings and duplicates
    return set([span for span in out if span.strip()])


## Triple Scoring

def extract_triples(annotation, pol_oversampling=7, neg_undersampling=0.7, ellipsis_oversampling=3):
    turns = annotation['tokens']
    triple_ids = annotation['annotations']

    arguments = defaultdict(list)
    triples = []
    labels = []

    triple_ids = [t[:4] for t in triple_ids]

    # Oversampling of elliptical triples
    for triple in deepcopy(triple_ids):
        turn_ids = set([i for i, _ in triple[0]]) | set([i for i, _ in triple[2]])
        if len(turn_ids) > 1:
            triple_ids += [triple] * ellipsis_oversampling

    for subj, pred, obj, neg in triple_ids:

        # Extract tokens belonging to triple arguments
        subj = ' '.join(turns[i][j] for i, j in subj) if subj else ''
        pred = ' '.join(turns[i][j] for i, j in pred) if pred else ''
        obj = ' '.join(turns[i][j] for i, j in obj) if obj else ''

        if subj or pred or obj:  # No blank triples

            if not neg:
                triples += [(subj, pred, obj)]
                labels += [1]
            else:
                triples += [(subj, pred, obj)] * pol_oversampling # oversampling negative polarities
                labels += [2] * pol_oversampling

            arguments['subjs'].append(subj)
            arguments['preds'].append(pred)
            arguments['objs'].append(obj)

    if not triples:
        return [], [], []

    # Create negative examples (i.e. not entailed)
    n = int(len(triples) * neg_undersampling)
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

    return turns, triples, labels