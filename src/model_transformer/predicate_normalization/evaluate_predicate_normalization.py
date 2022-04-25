from predicate_normalization import PredicateNormalizer
from sklearn.metrics import classification_report


def speaker_to_pronoun(arg, slot='subj'):
    """ Maps back from speaker ids to pronouns to help BERT with creating good
        embeddings (speaker* is not in the vocabulary).
    """
    if slot == 'subj':
        arg = arg.replace('speaker1', 'I').replace('speaker2', 'I')
    else:
        arg = arg.replace('speaker1', 'you').replace('speaker2', 'you')
    return arg.strip()


def load_test_data(fname):
    """ Extracts triples and corresponding normalized predicate
        labels from test set.
    """
    triples = []
    labels = []
    with open(fname, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                subj, pred, obj, _, norm_pred = line.strip().split(',')
                triples.append((speaker_to_pronoun(subj, slot='subj'),
                                speaker_to_pronoun(pred, slot='pred'),
                                speaker_to_pronoun(obj, slot='obj')))
                labels.append(norm_pred)
    return triples, labels


def evaluate(normalizer, test_triples, test_labels):
    """ Evaluates predicate normalization and disambiguation.
    """
    pred_labels = []
    for i, triple in enumerate(test_triples):
        norm_pred, conf = normalizer.normalize(*triple)
        pred_labels.append(norm_pred)
        print('%s (%s)' % (triple, conf))
        print('True:', norm_pred)
        print('Pred:', test_labels[i])
        print()

    print(classification_report(test_labels, pred_labels, zero_division=0))


if __name__ == '__main__':
    test_triples, test_labels = load_test_data('test_exemplars.txt')
    normalizer = PredicateNormalizer('canonical_exemplars.txt')
    evaluate(normalizer, test_triples, test_labels)
