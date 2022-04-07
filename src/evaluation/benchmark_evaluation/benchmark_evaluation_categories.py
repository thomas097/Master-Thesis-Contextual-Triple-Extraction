import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from run_transformer_pipeline import AlbertTripleExtractor
from baselines import ReVerbBaseline, OLLIEBaseline, StanfordOpenIEBaseline


def load_examples(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        dialogue, triple = None, None
        for line in file:
            if len(line.strip()) == 0:  # empty line -> new example
                data.append((dialogue, triple))
                dialogue, triple = None, None
            elif dialogue is None:
                dialogue = line.lower().strip()
            else:
                triple = tuple([arg.strip() for arg in line.lower().split(',')])

    # If no space at the end of the file, this one would be missed
    if dialogue is not None and triple is not None:
        data.append((dialogue, triple))

    return data


def evaluate(examples_file, model, threshold=0.5):
    # Where the triples found? Where there more we didn't want?
    recall = 0
    precision = 0

    # Extract triples from annotations
    examples = load_examples(examples_file)
    for i, (dialogue, expected_triple) in enumerate(examples):

        # Add <eos> if dialogue <3 turns
        if dialogue.count('<eos>') < 3:
            n = 3 - dialogue.count('<eos>')
            dialogue = '<eos> ' * n + dialogue

        # Predict triples
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), dialogue))
        found_triples = [triple for ent, triple in model.extract_triples(dialogue, verbose=False) if ent > threshold]
        print('expected:', expected_triple)
        print('found:   ', found_triples)

        # Recall: was the expected triple found?
        recall += expected_triple in found_triples

        # Precision: was there more found that we didn't want to find?
        precision += not [t for t in found_triples if t != expected_triple]

    R = recall / len(examples)
    P = precision / len(examples)
    F1 = 2 * P * R / (P + R)
    print('\nrecall:   ', R)
    print('precision:', P)
    print('f-measure:', F1)


if __name__ == '__main__':
    MODEL = 'albert'

    if MODEL == 'reverb':
        model = ReVerbBaseline()
    elif MODEL == 'ollie':
        model = OLLIEBaseline()
    elif MODEL == 'stanford':
        model = StanfordOpenIEBaseline()
    elif MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/argument_extraction_albert-v2_06_04_2022_multi',
                                      '../../model_transformer/models/scorer_albert-v2_06_04_2022_multi')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('test_examples/answer_ellipsis.txt', model)
