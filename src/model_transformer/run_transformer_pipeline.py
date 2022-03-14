from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from utils import *

from itertools import product


class AlbertTripleExtractor:
    def __init__(self, arg_model_path, scorer_model_path, base_model='albert-base-v2'):
        self._arg_model = ArgumentExtraction(base_model, path=arg_model_path)
        self._scorer_model = TripleScoring(base_model, path=scorer_model_path)

    def extract_triples(self, inputs):
        y_subjs, y_preds, y_objs, subwords = self._arg_model.predict(inputs.split())
        subjs = bio_tags_to_tokens(subwords, y_subjs.T, one_hot=True)
        preds = bio_tags_to_tokens(subwords, y_preds.T, one_hot=True)
        objs = bio_tags_to_tokens(subwords, y_objs.T, one_hot=True)

        print('subjs: ', subjs)
        print('preds: ', preds)
        print('objs:  ', objs, '\n')

        # Score all possible triples
        triples = []
        for subj, pred, obj in product(subjs, preds, objs):
            y_hat = self._scorer_model.predict(inputs.split(), [subj, pred, obj])
            entailed = 1 - y_hat[0]
            polarity = 'positive' if y_hat[1] > y_hat[2] else 'negative'
            triples.append((entailed, (subj, pred, obj, polarity)))

        # Rank triples from high (entailed) to low (barely entailed)
        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj), pol


if __name__ == '__main__':
    model = AlbertTripleExtractor('models/argument_extraction_albert-v2_14_03_2022',
                                  'models/scorer_albert-v2_14_03_2022')
    # Test!
    example = '<eos> do you live in Italy ? <eos> no, i\'m not, i\'m in new york. <eos>'
    print("Input: ", example, '\n')
    for score, triple, polarity in model.extract_triples(example):
        print(score, triple, polarity)
