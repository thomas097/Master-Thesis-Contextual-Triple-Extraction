import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

import glob
import json
import numpy as np
from tqdm import tqdm

from run_transformer_pipeline import AlbertTripleExtractor
from baselines import ReVerbBaseline, OLLIEBaseline, OpenIEBaseline


def load_annotations(path, files=None):
    used_files = files if files is not None else glob.glob(path + '/*.json')
    for fname in used_files:
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not data['skipped']:
                yield data


def triples_from_annotation(ann):
    triple_lst = ann['annotations']
    tokens = ann['tokens']

    triples = []
    for triple in triple_lst:
        subj = ' '.join([tokens[i][j] for i, j in triple[0]]).replace(' [unk] ', ' ')
        pred = ' '.join([tokens[i][j] for i, j in triple[1]]).replace(' [unk] ', ' ')
        obj_ = ' '.join([tokens[i][j] for i, j in triple[2]]).replace(' [unk] ', ' ')
        polar = 'positive' if not triple[3] else 'negative'

        # Skip blank triples
        if subj or pred or obj_:
            triples.append((subj, pred, obj_, polar))
    return triples


def confusion_matrix(predicted_triples, labeled_triples):
    tp, tn, fp, fn = 0, 0, 0, 0

    # Map triples to strings to allow convenient matching
    preds = set([' '.join(triple) for triple in predicted_triples])
    labels = set([' '.join(triple) for triple in labeled_triples])

    # Build confusion table
    for triple in preds | labels:
        if triple in preds and triple in labels:
            tp += 1
        elif triple in preds and triple not in labels:
            fp += 1
        elif triple not in preds and triple in labels:
            fn += 1
        else:
            tn += 1
    return np.array([tp, fp, fn, tn])


def evaluate(annotation_file, model, decision_thres=0.7):
    # Measure True Positives, True Negatives, etc.
    conf_matrix = np.zeros(4)

    # Extract triples from annotations
    for ann in tqdm(load_annotations(annotation_file)):
        # Ground truth
        y_true = triples_from_annotation(ann)

        # Predict triple
        input_ = ' '.join([t for ts in ann['tokens'] for t in ts + ['<eos>']])
        print(input_)
        y_pred = [triple for ent, triple in model.extract_triples(input_) if ent > decision_thres]
        print(y_pred)

        # Update confusion table
        conf_matrix += confusion_matrix(y_pred, y_true)

    # Print performance metrics
    print('Aggregate confusion table')
    print(conf_matrix.reshape(2, 2))

    print()
    tp, fp, fn, tn = conf_matrix
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * precision * recall / (precision + recall)
    print('precision:', precision)
    print('recall:   ', recall)
    print('F1:       ', F1)



if __name__ == '__main__':
    MODEL = 'albert'

    if MODEL == 'reverb':
        model = ReVerbBaseline()
    elif MODEL == 'ollie':
        model = OLLIEBaseline()
    elif MODEL == 'openie':
        model = OpenIEBaseline()
    elif MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/argument_extraction_albert-v2_6_04_2022',
                                      '../../model_transformer/models/scorer_albert-v2_6_04_2022')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('annotations', model)
