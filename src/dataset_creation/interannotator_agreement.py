from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import jaccard_distance, masi_distance
from glob import glob
from pathlib import Path
import random
import json
from pprint import pprint


def load_annotations(fname):
    triples = []
    with open(fname, 'r', encoding='utf-8') as file:
        data = json.load(file)
        tokens = data['tokens']
        for subj, pred, obj, pol, _ in data['annotations']:
            subj = ' '.join([tokens[i][j] for i, j in subj])
            pred = ' '.join([tokens[i][j] for i, j in pred])
            obj_ = ' '.join([tokens[i][j] for i, j in obj])
            polar = 'negative' if pol else 'positive'
            if subj or pred or obj_:
                triples.append((subj, pred, obj_, polar))
    return triples


def coder_annotations(filenames):
    annots = {}
    for fname in filenames:
        id_ = Path(fname).stem
        annots[id_] = load_annotations(fname)
    return annots


def two_coder_agreement(ann1, ann2, distance=jaccard_distance):
    # Determine shared ids
    ids = set(ann1.keys()).intersection(set(ann2.keys()))

    # Create matrix with, annotators, and triple labels
    task_data = []
    for id_ in ids:
        task_data.append(('coder1', id_, frozenset(ann1[id_]))) # triples for dialogue with id=id_
        task_data.append(('coder2', id_, frozenset(ann1[id_])))

    # Compute annotator agreement
    task = AnnotationTask(distance=distance)
    task.load_array(task_data)
    # No chance correction needed as random agreement is very unlikely (in the order of 1e-6)
    return task.avg_Ao()


def total_agreement(annotators):
    alphas = []
    for i, ann1 in enumerate(annotators):
        for j, ann2 in enumerate(annotators[i + 1:]):
            # Check whether annotators marked same samples
            if not set(ann1.keys()).intersection(set(ann2.keys())):
                print('no overlap')
                continue

            # Compute agreement
            alpha = two_coder_agreement(ann1, ann2)
            alphas.append(alpha)

    # Return average pairwise overlaps
    return sum(alphas) / len(alphas)


if __name__ == '__main__':
    # Fake overlapping annotations for now
    total_files = random.sample(glob('../annotation_tool/annotations/*.json'), 20)
    ann1 = coder_annotations(total_files[:15])
    ann2 = coder_annotations(total_files[-15:]) # overlap of 5
    ann3 = coder_annotations(random.sample(glob('../annotation_tool/annotations/*.json'), 20))  # no overlap

    print(total_agreement([ann1, ann2, ann3]))
