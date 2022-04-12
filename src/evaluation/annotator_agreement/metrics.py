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
    # Defines a func to check whether two triples match approximately (one arg is subset of another)
    def is_match(triple1, triple2):
        for arg1, arg2 in zip(triple1, triple2):
            if set(arg1).isdisjoint(set(arg2)):
                return False
            sub1 = set(arg1).issubset(set(arg2))
            sub2 = set(arg2).issubset(set(arg1))
            if not sub1 and not sub2:
                return False
        return True

    # Checks if triple has an approximate match in another
    def is_in(triple1, triple_list):
        return bool([triple2 for triple2 in triple_list if is_match(triple1, triple2)])

    # Compute f1 score
    def f1(a, b):
        r = 0
        for arg in b:
            if is_in(arg, a):
                r += 1
        r = r / len(b)

        p = 0
        for arg in a:
            if is_in(arg, b):
                p += 1
        p = p / len(a)

        if p == 0 and r == 0:
            return 0

        return 2 * p * r / (p + r)

    return (f1(triples1, triples2) + f1(triples2, triples1)) / 2


def argument_jaccard(args1, args2):
    # Strip prepositions, determines and particles
    args1 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', ' '.join(t)) for t in args1}
    args2 = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', ' '.join(t)) for t in args2}

    # Jaccard = intersection / union
    return len(args1.intersection(args2)) / len(args1 | args2)


def argument_f1(args1, args2):
    # Strip prepositions, determines and particles
    a = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', ' '.join(t)) for t in args1}
    b = {re.sub(r'( |^)(a|the|\'|\.|\,) ', ' ', ' '.join(t)) for t in args2}
    return (f_measure(a, b) + f_measure(b, a)) / 2


def argument_soft_f1(args1, args2):
    # Defines a func to check whether two triples match approximately (one arg is subset of another)
    def is_match(arg1, arg2):
        if set(arg1).isdisjoint(set(arg2)):
            return False
        sub1 = set(arg1).issubset(set(arg2))
        sub2 = set(arg2).issubset(set(arg1))
        return sub1 or sub2

    # Checks if argument has an approximate match in another
    def is_in(arg, arg_list):
        return bool([arg_list for arg_list in arg_list if is_match(arg, arg_list)])

    # Compute f1 score
    def f1(a, b):
        r = 0
        for arg in b:
            if is_in(arg, a):
                r += 1
        r = r / len(b)

        p = 0
        for arg in a:
            if is_in(arg, b):
                p += 1
        p = p / len(a)

        if p == 0 and r == 0:
            return 0

        return 2 * p * r / (p + r)

    return (f1(args1, args2) + f1(args1, args2)) / 2

