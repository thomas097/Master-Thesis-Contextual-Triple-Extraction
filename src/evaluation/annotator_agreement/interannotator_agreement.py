import krippendorff
from metrics import *
from utils import *
import pandas as pd


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


def pairwise_argument_agreement(path, arg, metric='F1', corrected=False):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_annotators = list()
    for fname in sorted(glob.glob(path + '/*')):
        annotator = Path(fname).parts[-1]
        annotations[annotator] = load_annotations(fname)
        all_annotators.append(annotator)

    # Compute pairwise agreement between annotators
    all_annotators = sorted(list(annotations.keys()))
    pairwise_mat = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.float32)
    pairwise_cnt = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.float32)

    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlap = files1.intersection(files2)

            if overlap:
                # Compute triple overlap on each file and annotator pair (that annotated the file)
                for file in overlap:
                    args1 = arguments(annotations[ann1][file], arg=arg, corrected=corrected)
                    args2 = arguments(annotations[ann2][file], arg=arg, corrected=corrected)

                    if not args1 and not args2:
                        continue

                    if metric == 'jaccard':
                        pairwise_mat[i, j] += argument_jaccard(args1, args2)
                        pairwise_cnt[i, j] += 1
                    elif metric == 'F1':
                        pairwise_mat[i, j] += argument_f1(args1, args2)
                        pairwise_cnt[i, j] += 1
                    else:
                        raise Exception('no metric called', metric)

    # Fix divide by zero if no overlap
    pairwise_cnt[pairwise_cnt == 0] = 1

    # Print results
    all_annotators = [t.replace('annotations_', '') for t in all_annotators]
    print('Pairwise agreement %s: corrected=%s' % (arg, corrected))
    df = pd.DataFrame(pairwise_mat / pairwise_cnt, columns=all_annotators, index=all_annotators).round(3)
    print(df)
    print()


def pairwise_triple_agreement(path, metric='F1', corrected=False):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_annotators = list()
    for fname in sorted(glob.glob(path + '/*')):
        annotator = Path(fname).parts[-1]
        annotations[annotator] = load_annotations(fname)
        all_annotators.append(annotator)

    # Compute pairwise agreement between annotators
    all_annotators = sorted(list(annotations.keys()))
    pairwise_mat = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.float32)
    pairwise_cnt = np.zeros((len(all_annotators), len(all_annotators)), dtype=np.float32)

    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlap = files1.intersection(files2)

            if overlap:
                # Compute triple overlap on each file and annotator pair (that annotated the file)
                for file in overlap:
                    triples1 = triples(annotations[ann1][file], corrected=corrected)
                    triples2 = triples(annotations[ann2][file], corrected=corrected)

                    if not triples1 and not triples2:
                        continue

                    if metric == 'jaccard':
                        pairwise_mat[i, j] += triple_jaccard(triples1, triples2)
                    elif metric == 'F1':
                        pairwise_mat[i, j] += triple_f1(triples1, triples2)
                    elif metric == 'soft_F1':
                        pairwise_mat[i, j] += triple_soft_f1(triples1, triples2)
                    else:
                        raise Exception('no metric called', metric)
                    pairwise_cnt[i, j] += 1

    # Fix divide by zero if no overlap
    pairwise_cnt[pairwise_cnt == 0] = 1

    # Print results
    all_annotators = [t.replace('annotations_', '') for t in all_annotators]
    print('Pairwise agreement triples: corrected=%s metric=%s' % (corrected, metric))
    df = pd.DataFrame(pairwise_mat / pairwise_cnt, columns=all_annotators, index=all_annotators).round(3)
    print(df)
    print()


def avg_triple_agreement(path, metric='jaccard', corrected=False):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_annotators = list()
    for fname in sorted(glob.glob(path + '/*')):
        annotator = Path(fname).parts[-1]
        annotations[annotator] = load_annotations(fname)
        all_annotators.append(annotator)

    # Compute average agreement over all files
    scores = []
    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlapping_files = files1.intersection(files2)

            if overlapping_files and i != j:
                # Compute triple overlap on each file and annotator pair (that annotated the file)
                for file in overlapping_files:
                    triples1 = triples(annotations[ann1][file], corrected=corrected)
                    triples2 = triples(annotations[ann2][file], corrected=corrected)

                    if not triples1 and not triples2:
                        continue

                    if metric == 'jaccard':
                        scores.append(triple_jaccard(triples1, triples2))
                    elif metric == 'F1':
                        scores.append(triple_f1(triples1, triples2))
                    elif metric == 'soft_F1':
                        scores.append(triple_soft_f1(triples1, triples2))
                    else:
                        raise Exception('no metric called', metric)

    print("Avg. pairwise agreement triples (%s): %s" % (metric, np.mean(scores)))


def avg_argument_agreement(path, arg='subj', metric='jaccard', corrected=False):
    # Load each annotator's annotations and convert to bin vector of token assignments to arg
    annotations = dict()
    all_annotators = list()
    for fname in sorted(glob.glob(path + '/*')):
        annotator = Path(fname).parts[-1]
        annotations[annotator] = load_annotations(fname)
        all_annotators.append(annotator)

    # Compute average agreement over all files
    scores = []
    for i, ann1 in enumerate(all_annotators):
        for j, ann2 in enumerate(all_annotators):

            # Do they have overlapping annotations?
            files1 = set(annotations[ann1].keys())
            files2 = set(annotations[ann2].keys())
            overlap = files1.intersection(files2)

            if overlap and i != j:
                # Compute triple overlap on each file and annotator pair (that annotated the file)
                for file in overlap:
                    args1 = arguments(annotations[ann1][file], arg=arg, corrected=corrected)
                    args2 = arguments(annotations[ann2][file], arg=arg, corrected=corrected)

                    if not args1 and not args2:
                        continue

                    if metric == 'jaccard':
                        scores.append(argument_jaccard(args1, args2))
                    elif metric == 'F1':
                        scores.append(argument_f1(args1, args2))
                    else:
                        raise Exception('no metric called', metric)

    print("Avg. pairwise agreement of %s (%s): %s" % (arg, metric, np.mean(scores)))



if __name__ == '__main__':
    print('\n##### Arguments #####\n')
    print('\n## average ##\n')
    avg_argument_agreement('annotations', arg='subj', metric='F1', corrected=True)
    avg_argument_agreement('annotations', arg='pred', metric='F1', corrected=True)
    avg_argument_agreement('annotations', arg='obj', metric='F1', corrected=True)

    print('\n## pairwise ##\n')
    pairwise_argument_agreement('annotations', arg='subj', metric='F1', corrected=True)
    pairwise_argument_agreement('annotations', arg='pred', metric='F1', corrected=True)
    pairwise_argument_agreement('annotations', arg='obj', metric='F1', corrected=True)

    print('\n##### Triples #####\n')
    print('\n## average ##\n')
    avg_triple_agreement('annotations', metric='jaccard', corrected=True)
    avg_triple_agreement('annotations', metric='F1', corrected=True)
    avg_triple_agreement('annotations', metric='soft_F1', corrected=True)

    print('\n## pairwise ##\n')
    pairwise_triple_agreement('annotations', metric='jaccard', corrected=True)
    pairwise_triple_agreement('annotations', metric='F1', corrected=True)
    pairwise_triple_agreement('annotations', metric='soft_F1', corrected=True)




