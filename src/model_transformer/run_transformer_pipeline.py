from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from utils import *

from itertools import product


class AlbertTripleExtractor:
    def __init__(self, arg_model_path, scorer_model_path, base_model='albert-base-v2'):
        self._arg_model = ArgumentExtraction(base_model, path=arg_model_path)
        self._scorer_model = TripleScoring(base_model, path=scorer_model_path)

    def extract_triples(self, inputs, confidenec=0.4):
        y_subjs, y_preds, y_objs, subwords = self._arg_model.predict(inputs.split())
        subjs = bio_tags_to_tokens(subwords, y_subjs.T, one_hot=True)
        preds = bio_tags_to_tokens(subwords, y_preds.T, one_hot=True)
        objs = bio_tags_to_tokens(subwords, y_objs.T, one_hot=True)

        print('subjs: ', subjs)
        print('preds: ', preds)
        print('objs:  ', objs, '\n')

        # Score all possible triples
        triple_scores = []
        for subj, pred, obj in product(subjs, preds, objs):
            entailed, polarity = self._scorer_model.predict(inputs.split(), [subj, pred, obj])
            triple_scores.append((entailed[1], (subj, pred, obj, polarity[1])))

        # Rank triples from high (entailed) to low (not entailed)
        for entailed, (subj, pred, obj, pol) in sorted(triple_scores, key=lambda x: -x[0]):
            if entailed > confidenec:
                yield entailed, (subj, pred, obj), pol


if __name__ == '__main__':
    model = AlbertTripleExtractor('models/argument_extraction_albert-v2_09_03_2022',
                                  'models/scorer_albert-v2_09_03_2022')
    # Test!
    example = 'I do n\'t like animals <eos>'
    print("Input: ", example, '\n')
    for score, triple, polarity_score in model.extract_triples(example):
        polarity = 'positive' if polarity_score > 0.5 else 'negative'
        print(score, triple, polarity)
