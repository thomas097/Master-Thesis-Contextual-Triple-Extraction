import json
import glob
import numpy as np
from pathlib import Path
import re
import krippendorff


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


def triples(annotation):
    tokens = annotation['tokens']
    triples = []
    for subj, pred, obj, pol, cert in annotation['annotations']:
        subj = [tokens[i][j] for i, j in subj]
        pred = [tokens[i][j] for i, j in pred]
        obj = [tokens[i][j] for i, j in obj]
        triples.append((subj, pred, obj))
    return triples


def average_krippendorff_token_assignments(path, arg):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_files = set()
    for fname in glob.glob(path + '/*'):
        annotator = Path(fname).parts[-1]
        token_assignments = {file: token_argument_assignments(ann, arg) for file, ann in
                             load_annotations(fname).items()}
        annotations[annotator] = token_assignments

        # Record list of all files annotated
        all_files |= set(token_assignments.keys())

    # Compute average agreement over all files
    alphas = []
    for file in all_files:
        # Which annotators annotated this file
        annotators_with_file = [ann for ann in annotations if file in annotations[ann]]

        if len(annotators_with_file) > 1:

            # Create MxN reliability matrix (M = #annotators, N = #tokens in dialogue)
            rel_mat = np.array([annotations[ann][file] for ann in annotators_with_file])

            # Compute Krippendorff alpha over file
            alpha = krippendorff.alpha(rel_mat, level_of_measurement='nominal')
            alphas.append(alpha)
    print("Avg. agreement %ss: %s" % (arg, np.mean(alphas)))


def pairwise_krippendorff_token_assignments(path, arg):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_files = set()
    for fname in glob.glob(path + '/*'):
        annotator = Path(fname).parts[-1]
        token_assignments = {file: token_argument_assignments(ann, arg) for file, ann in
                             load_annotations(fname).items()}
        annotations[annotator] = token_assignments

        # Record list of all files annotated
        all_files |= set(token_assignments.keys())

    # Compute pairwise agreement between annotators
    all_annotators = sorted(list(annotations.keys()))
    pairwise_mat = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.float32)
    pairwise_cnt = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.uint8)

    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlap = files1.intersection(files2)

            if overlap:
                # Create reliability matrix
                ann1_annotations = [annotations[ann1][file] for file in overlap]
                ann2_annotations = [annotations[ann2][file] for file in overlap]
                rel_mat = np.array([[x for xs in ann1_annotations for x in xs],
                                    [x for xs in ann2_annotations for x in xs]])

                # Compute Krippendorff alpha over two annotators
                alpha = krippendorff.alpha(rel_mat, level_of_measurement='nominal')

                pairwise_mat[i, j] += alpha
                pairwise_cnt[i, j] += 1

    # Fix divide by zero
    pairwise_cnt[pairwise_cnt == 0] = 1

    # Print results
    print('Pairwise agreement %s:' % arg)
    print(pairwise_mat / pairwise_cnt)
    print()


def triple_jaccard(triples1, triples2):
    # Format triples as strings, e.g. "I | like to go to | the gym"
    triples1 = [' | '.join([' '.join(arg) for arg in triple]) for triple in triples1]
    triples2 = [' | '.join([' '.join(arg) for arg in triple]) for triple in triples2]

    # Strip prepositions, determines and particles
    triples1 = {re.sub(r' (a|the|to|at|of|\'|in|\.|\,) ', ' ', t) for t in triples1}
    triples2 = {re.sub(r' (a|the|to|at|of|\'|in|\.|\,) ', ' ', t) for t in triples2}

    # Jaccard = intersection / union
    return len(triples1.intersection(triples2)) / len(triples1 | triples2)


def raw_triple_agreement(path):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_annotators = list()
    for fname in sorted(glob.glob(path + '/*')):
        annotator = Path(fname).parts[-1]
        annotations[annotator] = load_annotations(fname)
        all_annotators.append(annotator)

    # Compute average agreement over all files
    jaccard_dists = []
    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlap = files1.intersection(files2)

            if overlap and i != j:
                # Compute triple overlap on each file and annotator pair (that annotated the file)
                for file in overlap:
                    triples1 = triples(annotations[ann1][file])
                    triples2 = triples(annotations[ann2][file])
                    jaccard_dists.append(triple_jaccard(triples1, triples2))

    print("Mean pairwise Jaccard index of triple annotations: %s" % np.mean(jaccard_dists))


if __name__ == '__main__':
    print('\n##### Krippendorff alpha for token assignments #####\n')
    average_krippendorff_token_assignments('annotations', arg='subj')
    average_krippendorff_token_assignments('annotations', arg='pred')
    average_krippendorff_token_assignments('annotations', arg='obj')

    print('\n##### Pairwise agreement between annotators #####\n')
    pairwise_krippendorff_token_assignments('annotations', arg='subj')
    pairwise_krippendorff_token_assignments('annotations', arg='pred')
    pairwise_krippendorff_token_assignments('annotations', arg='obj')

    print('\n##### Raw triple agreement #####\n')
    raw_triple_agreement('annotations')




