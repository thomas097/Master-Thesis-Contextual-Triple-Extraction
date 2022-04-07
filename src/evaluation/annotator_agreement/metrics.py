import re
from nltk.metrics.scores import f_measure


def triple_jaccard(triples1, triples2):
    # Format triples as strings, e.g. "I | like to go to | the gym"
    triples1 = [' | '.join([' '.join(arg) for arg in triple]) for triple in triples1]
    triples2 = [' | '.join([' '.join(arg) for arg in triple]) for triple in triples2]

    # Strip prepositions, determines and particles
    triples1 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in triples1}
    triples2 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in triples2}

    # Jaccard = intersection / union
    return len(triples1.intersection(triples2)) / len(triples1 | triples2)


def triple_f1(triples1, triples2):
    # Format triples as strings, e.g. "I | like to go to | the gym"
    a = set([' | '.join([' '.join(arg) for arg in triple]) for triple in triples1])
    b = set([' | '.join([' '.join(arg) for arg in triple]) for triple in triples2])
    return (f_measure(a, b) + f_measure(b, a)) / 2


def triple_soft_f1(triples1, triples2):
    # Defines a func to check whether two triples match approximately (at least 2 arguments)
    def is_match(triple1, triple2):
        return sum([int(triple1[i] == triple2[i]) for i in range(3)]) >= 2

    # Checks if triple has an approximate match in another
    def is_in(triple1, triple_list):
        return bool([t for t in triple_list if is_match(triple1, t)])

    # Compute f1 score
    def f1(a, b):
        tp, fp, fn = 0, 0, 0
        for t in a + b:
            if is_in(t, a) and is_in(t, b):
                tp += 1
            elif not is_in(t, a) and is_in(t, b):
                fn += 1
            elif is_in(t, a) and not is_in(t, b):
                fp += 1

        if tp == 0:
            return 0

        r = tp / (tp + fn)
        p = tp / (tp + fp)
        return 2 * p * r / (p + r)

    return (f1(triples1, triples2) + f1(triples2, triples1)) / 2


def argument_jaccard(args1, args2):
    # Strip prepositions, determines and particles
    args1 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in args1}
    args2 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in args2}

    # Jaccard = intersection / union
    return len(args1.intersection(args2)) / len(args1 | args2)


def argument_f1(args1, args2):
    # Strip prepositions, determines and particles
    a = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in args1}
    b = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', t) for t in args2}
    return (f_measure(a, b) + f_measure(b, a)) / 2

