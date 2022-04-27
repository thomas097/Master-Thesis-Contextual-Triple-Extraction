from run_transformer_pipeline import AlbertTripleExtractor
from LeolaniTripleExtraction import LeolaniBaseline
from OLLIE import OllieBaseline
from OpenIE5 import OpenIE5Baseline
from StanfordOpenIE import StanfordOpenIEBaseline
from metrics import classification_report
import spacy


def load_examples(path):
    """ Load examples in the form of (str: dialogue, list: triples).

    :param path: Path to test file (e.g. 'test_examples/test_full.txt')
    :return:     List of (str: dialogue, list: triples) pairs.
    """
    # Extract lines corresponding to dialogs and triples
    samples = []
    with open(path, 'r', encoding='utf-8') as file:
        block = []
        for line in file:
            if line.strip() and not line.startswith('#'):
                block.append(line.strip())
            elif block:
                samples.append(block)
                block = []
        samples.append(block)

    # Split triple arguments
    examples = []
    for block in samples:
        dialog = block[1]
        triples = [string_to_triple(triple) for triple in block[2:]]
        examples.append((dialog, triples))

    return examples


def string_to_triple(text_triple):
    """ Tokenizes triple line into individual arguments

    :param text_triple: plain-text string of triple
    :return:            triple of the form (subj, pred, obj, polar)
    """
    return tuple([x.strip() for x in text_triple.split(',')])


def lemmatize_triple(subj, pred, obj, polar, nlp):
    subj = ' '.join([t.lemma_ for t in nlp(subj)])
    pred = ' '.join([t.lemma_ for t in nlp(pred)])
    obj = ' '.join([t.lemma_ for t in nlp(obj)])
    return subj, pred, obj, polar


def evaluate(test_file, model, num_samples=-1, k=0.9, lemmatize=True):
    """ Evaluates the model on a test file, yielding scores for precision@k,
        recall@k, F1@k and PR-AUC.

    :param test_file:   Test file from '/test_examples'
    :param model:       Albert, Dependency or baseline model instance
    :param num_samples: The maximum number of samples to evaluate (default: all)
    :param k:           Confidence level at which to evaluate models
    :param lemmatize:   Whether to lemmatize predicates to make sure duplicate predicates such as "is"
                        and "are" are removed and match across baselines (default: True)
    :return:            Scores for precision@k, recall@k, F1@k and PR-AUC
    """
    # Extract dialog-triples pairs from annotations
    examples = load_examples(test_file)
    if num_samples > 0:
        examples = examples[:num_samples]

    # Predictions
    true_triples = []
    pred_triples = []
    for i, (text, triples) in enumerate(examples):
        # Print progress
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), text))

        # Predict triples
        extractions = model.extract_triples(text, verbose=True)

        # Strip negation/certainty of not in test set
        nx = min([len(t) for t in triples])
        extractions = set([t[:nx] for t in extractions])

        print('expected:', triples)
        print('found:   ', [t for c, t in extractions if c > k])

        true_triples.append(triples)
        pred_triples.append(extractions)

    # If lemmatize is enabled, map word forms to lemmas
    if lemmatize:
        nlp = spacy.load('en_core_web_sm')
        true_triples = [set([lemmatize_triple(*triple, nlp) for triple in lst]) for lst in true_triples]
        pred_triples = [set([(conf, lemmatize_triple(*triple, nlp)) for conf, triple in lst]) for lst in pred_triples]

    # Compute performance metrics
    return classification_report(true_triples, pred_triples, k=k)


if __name__ == '__main__':
    MODEL = 'ollie'

    if MODEL == 'openie5':
        model = OpenIE5Baseline()
    elif MODEL == 'ollie':
        model = OllieBaseline()
    elif MODEL == 'stanford':
        model = StanfordOpenIEBaseline()
    elif MODEL == 'leolani':
        model = LeolaniBaseline()
    elif MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/2022-04-11', speaker1='speaker1', speaker2='speaker2')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('test_examples/test_declarative_statements.txt', model, num_samples=5)
