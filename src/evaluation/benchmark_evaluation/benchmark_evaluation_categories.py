import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

import numpy as np
import spacy
from tqdm import tqdm
from run_transformer_pipeline import AlbertTripleExtractor
from baselines import ReVerbBaseline, OLLIEBaseline, OpenIEBaseline


def load_examples(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        dialogue, triple = None, None
        for line in file:
            if len(line.strip()) == 0: # empty line -> reset
                data.append((dialogue, triple))
                dialogue, triple = None, None
            elif dialogue is None:
                dialogue = line
            else:
                triple = line
    return data


def evaluate(examples_file, model, threshold=0.7):
    # Extract triples from annotations
    examples = load_examples(examples_file)
    for i, (dialogue, triple) in enumerate(examples):

        # Add <eos> if dialogue <3 turns
        if dialogue.count('<eos>') < 3:
            n = 3 - dialogue.count('<eos>')
            dialogue = '<eos> ' * n + dialogue.strip()

        # Predict triples
        print('\%s/%s input: %s' % (i + 1, len(examples), dialogue))
        found_triples = [triple for ent, triple in model.extract_triples(dialogue, verbose=False) if ent > threshold]
        print(found_triples)


if __name__ == '__main__':
    MODEL = 'albert'

    if MODEL == 'reverb':
        model = ReVerbBaseline()
    elif MODEL == 'ollie':
        model = OLLIEBaseline()
    elif MODEL == 'openie':
        model = OpenIEBaseline()
    elif MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/argument_extraction_albert-v2_06_04_2022_multi',
                                      '../../model_transformer/models/scorer_albert-v2_06_04_2022_multi')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('test_examples/simple_statements.txt', model)
