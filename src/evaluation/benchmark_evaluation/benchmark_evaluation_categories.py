import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from run_transformer_pipeline import AlbertTripleExtractor
from LeolaniTripleExtraction import LeolaniBaseline
from OLLIE import OLLIEBaseline
from OpenIE5 import OpenIE5Baseline
from StanfordOpenIE import StanfordOpenIEBaseline


def load_examples(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        block = []
        for line in file:
            if line.strip() == '':
                text = block[1]
                triples = [string_to_triple(t) for t in block[2:]]
                data.append((text, triples))
                block = []
            else:
                block.append(line.strip())

        if block:
            text = block[1]
            triples = [string_to_triple(t) for t in block[2:]]
            data.append((text, triples))
    return data


def string_to_triple(triple):
    return tuple([x.strip() for x in triple.split(',')])


def evaluate(examples_file, model, num_samples=1000000, threshold=0.9):
    # Where the triples found? Where there more we didn't want?
    recall = 0
    total = 0

    # Extract triples from annotations
    examples = load_examples(examples_file)[:num_samples]

    for i, (text, exp_triples) in enumerate(examples):

        # Predict triples
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), text))
        found_triples = [triple for ent, triple in model.extract_triples(text, verbose=True) if ent > threshold]

        # Strip negation/certainty of not in test set
        nx = len(exp_triples[0])
        found_triples = [t[:nx] for t in found_triples]

        print('expected:', exp_triples)
        print('found:   ', found_triples)

        # Recall: was the expected triple found?
        for exp_triple in exp_triples:
            if exp_triple in found_triples:
                recall += 1
                print('+', exp_triple)
            else:
                print('-', exp_triple)
            total += 1

    # Performance scores
    R = recall / total
    print('\nrecall:   ', R)


if __name__ == '__main__':
    MODEL = 'albert'

    if MODEL == 'openie5':
        model = OpenIE5Baseline()
    elif MODEL == 'ollie':
        model = OLLIEBaseline()
    elif MODEL == 'stanford':
        model = StanfordOpenIEBaseline()
    elif MODEL == 'leolani':
        model = LeolaniBaseline()
    elif MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/2022-04-11', speaker1='speaker1', speaker2='speaker2')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('test_examples/test_yes_answers.txt', model, num_samples=30)
