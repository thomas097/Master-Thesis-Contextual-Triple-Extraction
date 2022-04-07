from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from utils import *
import spacy

from itertools import product


class AlbertTripleExtractor:
    def __init__(self, arg_model_path, scorer_model_path, base_model='albert-base-v2', sep='<eos>'):
        self._arg_model = ArgumentExtraction(base_model, path=arg_model_path)
        self._scorer_model = TripleScoring(base_model, path=scorer_model_path)
        self._nlp = spacy.load('en_core_web_sm')
        self._sep = sep

    def _tokenize(self, inputs):
        # Tokenize each turn separately (and splitting "n't" off)
        tokens = []
        for turn in inputs.split(self._sep):
            turn = turn.strip()
            if turn:
                tokens += [t.lower_ for t in self._nlp(turn)] + ['<eos>']
        return tokens

    def extract_triples(self, inputs, verbose=True):
        # 1. Extract SPO arguments from token sequence
        y_subjs, y_preds, y_objs, subwords = self._arg_model.predict(self._tokenize(inputs))
        subjs = bio_tags_to_tokens(subwords, y_subjs.T, one_hot=True)
        preds = bio_tags_to_tokens(subwords, y_preds.T, one_hot=True)
        objs = bio_tags_to_tokens(subwords, y_objs.T, one_hot=True)

        # 1b. add additional verbs to predicate using SpaCy
        for turn in inputs.split(self._sep):
            turn_tokens = self._nlp(turn)
            subjs |= {t.lower_ for t in turn_tokens if t.pos_ in ['NOUN', 'ADJ', 'PROPN']}
            preds |= {t.lower_ for t in turn_tokens if t.pos_ in ['VERB', 'AUX']}
            objs |= {t.lower_ for t in turn_tokens if t.pos_ in ['NOUN', 'ADJ', 'PROPN']}

        if verbose:
            print('subjs: ', subjs)
            print('preds: ', preds)
            print('objs:  ', objs, '\n')

        # 2. Compute all possible triples
        candidates = [list(triple) for triple in product(subjs, preds, objs)]

        # 3. Score candidate triples
        predictions = self._scorer_model.predict_multi(inputs.split(), candidates)

        # 4. Rank candidates according to entailment predictions
        triples = []
        for y_hat, (subj, pred, obj) in zip(predictions, candidates):
            pol = 'negative' if y_hat[2] > y_hat[1] else 'positive'
            ent = max(y_hat[1], y_hat[2])
            triples.append((ent, (subj, pred, obj, pol)))

        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj, pol)


if __name__ == '__main__':
    model = AlbertTripleExtractor('models/argument_extraction_albert-v2_06_04_2022_multi',
                                  'models/scorer_albert-v2_06_04_2022_multi')
    # Test!
    example = 'i enjoy watching sports but do not want to do homework <eos> what do you like ? <eos> animals , but not cats <eos>'

    print("Input: ", example, '\n')
    for score, triple in model.extract_triples(example):
        print(score, triple)
