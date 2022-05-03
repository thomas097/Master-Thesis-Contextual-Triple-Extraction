import matplotlib.pyplot as plt
import numpy as np
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import auc


def is_in(x, lst, k=3):
    """ Performs a soft matching to get rid of tokenization artifacts. """
    for y in lst:
        mean_dist = np.max([levenshtein_distance(x[i], y[i]) for i in range(len(x))])
        if mean_dist <= k:
            return True
    return False


def precision_at_k(y_true, y_pred, k):
    tp, fp, = 0, 0
    for triples_true, triples_pred in zip(y_true, y_pred):
        # Filter out triples below k
        triples_pred = set([triple for conf, triple in triples_pred if conf >= k])

        for triple in triples_pred:
            if is_in(triple, triples_true):
                tp += 1
            else:
                fp += 1

    # Catch divide by zero
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall_at_k(y_true, y_pred, k):
    tp, fn, = 0, 0
    for triples_true, triples_pred in zip(y_true, y_pred):
        # Filter out triples below k
        triples_pred = set([triple for conf, triple in triples_pred if conf >= k])

        for triple in triples_true:
            if is_in(triple, triples_pred):
                tp += 1
            else:
                fn += 1

    # Catch divide by zero
    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def f_score_at_k(y_true, y_pred, k):
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def precision_recall_auc(y_true, y_pred, steps=2000, plot_pr=True):
    precision, recall = [], []
    for k in np.linspace(0, 0.999, steps):
        precision.append(precision_at_k(y_true, y_pred, k))
        recall.append(recall_at_k(y_true, y_pred, k))

    # Ensure complete curve
    precision.append(1)
    recall.append(0)

    if plot_pr:
        plt.grid('on', c='lightgrey')
        plt.plot([0, 1], [1, 0], 'r--', linewidth=1)
        plt.plot(recall, precision)
        plt.xlabel('recall (R)')
        plt.ylabel('precision (P)')
        plt.show()

    return auc(recall, precision)


def classification_report(true_triples, pred_triples, k):
    """ Computes precision@k, recall@k, F1@k and PR-AUC over gold
        triples and predictions.

    :param true_triples:
    :param pred_triples:
    :param k:
    :return:
    """
    precision = precision_at_k(true_triples, pred_triples, k)
    recall = recall_at_k(true_triples, pred_triples, k)
    fscore = f_score_at_k(true_triples, pred_triples, k)
    auc = precision_recall_auc(true_triples, pred_triples)

    print('precision@k:', precision)
    print('recall@k:   ', recall)
    print('F-score@k:  ', fscore)
    print('AUC:        ', auc)
    return precision, recall, fscore, auc


if __name__ == '__main__':
    """ 
        sent1: I like cats
        sent2: I despise aliens
        sent3: I went for a walk
        sent4: Candy are not good and bad
        sent5: You are a hyper like my cats
    """
    true_triples = [[("I", "like", "cats")],
                    [("I", "despise", "aliens")],
                    [("I", "went for", "a walk")],
                    [("candy", "are", "good"), ("candy", "are", "bad")],
                    [("you", "are", "hyper"), ("my cats", "are", "hyper")]]

    pred_triples = [[(0.98, ("I", "like", "cats"))],
                    [(0.94, ("I", "despise", "aliens"))],
                    [(0.9, ("I", "went for", "a walk"))],
                    [(0.6, ("candy", "are", "bad")), (0.82, ("candy", "are", "good")), (0.864, ("candy", "is", "good"))],
                    [(0.85, ("you", "are", "hyper")), (0.73, ("my cats", "are", "hyper")), (0.7, ("candy", "are", "good"))]]

    print('precision@k:', precision_at_k(true_triples, pred_triples, k=0.0))
    print('recall@k:   ', recall_at_k(true_triples, pred_triples, k=0.0))
    print('F-score@k:  ', f_score_at_k(true_triples, pred_triples, k=0.0))
    print('AUC:        ', precision_recall_auc(true_triples, pred_triples))