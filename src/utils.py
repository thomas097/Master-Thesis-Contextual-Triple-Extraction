import glob
import json
import random
import itertools
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
    triples = annotation['triples']
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

        if pred == 1:  # Beginning
            out.append(' '.join(span))
            span = [token.replace('Ġ', '')]

        elif pred == 2:  # Inside
            span.append(token.replace('Ġ', ''))

    if span:
        out.append(' '.join(span))
    return set(out[1:])


## Triple Scoring

def extract_triples(annotation):
    """ Extracts triples from the annotations.
    """
    turns = annotation['tokens']
    triples = annotation['triples']
    num_tokens = sum([len(turn) for turn in turns])

    triple_tokens = []
    for triple in triples:
        arg_tokens = [[], [], []]
        for k, arg in enumerate(triple[:3]):
            if not arg:
                arg_tokens[k].append('')
            for i, j in arg:
                token = turns[i][j]
                arg_tokens[k].append(token)
        triple_tokens.append([' '.join(a) for a in arg_tokens if a])

    return [t for t in triple_tokens if t != ['', '', '']]


def extract_negative_triples(triples):
    """ Creates negative examples from ground-truth triples.
    """
    # Group all subjects, all objects, and so on.
    subjs, preds, objs = [],[],[]
    for subj, pred, obj in triples:
        subjs.append(subj)
        preds.append(pred)
        objs.append(obj)

    # Sample random triples not part of the true examples
    fake_triples = []
    for subj, pred, obj in itertools.product(subjs, preds, objs):
        triple = [subj, pred, obj]
        if triple not in triples and triple not in fake_triples: # no duplicates
            fake_triples.append(triple)

    # Select a random sample (to avoid imbalance)
    random.shuffle(fake_triples)
    return fake_triples[:len(triples)]


## Co-reference Resolution

def load_visual_pcr(path):
    turns, labels = [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line.replace('true', 'True').replace('false', 'False'))

            dialog = [t for ts in data['sentences'] for t in ts]

            if data['pronoun_info']:
                pronoun_info = data['pronoun_info'][0]

                i, j = pronoun_info['current_pronoun']
                pronoun = ' '.join(dialog[i:j + 1])

                if pronoun_info['correct_NPs']:
                    k, l = pronoun_info['correct_NPs'][0]
                    target = ' '.join(dialog[k:l + 1])

                    idx = [idx for idx, sent in enumerate(data['sentences']) if pronoun in sent][0]
                    sents = data['sentences'][:idx + 1]

                    if target.lower() not in ' '.join(sents[0]).lower():  # first is caption
                        turns.append([' '.join(s).lower() for s in sents[1:]])
                        labels.append((pronoun, target))

    turns, labels = zip(*[(x, y) for x, y in zip(turns, labels) if len(x)])
    return turns, labels
