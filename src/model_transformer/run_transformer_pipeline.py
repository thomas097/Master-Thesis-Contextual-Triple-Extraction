from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from utils import *
import spacy

from itertools import product


class AlbertTripleExtractor:
    def __init__(self, arg_model_path, scorer_model_path, base_model='albert-base-v2', sep='<eos>',
                 speaker1='you', speaker2='I'):
        self._argument_module = ArgumentExtraction(base_model, path=arg_model_path)
        self._scoring_module = TripleScoring(base_model, path=scorer_model_path)
        self._nlp = spacy.load('en_core_web_sm')
        self._sep = sep

        # Assign identities to speakers
        self._speaker1 = speaker1
        self._speaker2 = speaker2

    def _tokenize(self, inputs):
        # Tokenize each turn separately (and splitting "n't" off)
        tokens = []
        for turn in inputs.split(self._sep):
            turn = turn.strip().lower()
            if turn:
                tokens += [t.lower_ for t in self._nlp(turn)] + ['<eos>']
        return tokens

    def _disambiguate_identity(self, token, turn_idx):
        # Even turns -> speaker1
        if turn_idx % 2 == 0:
            if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
                return self._speaker1
            elif token in ['my', 'mine', 'our', 'ours']:
                return self._speaker1 + "'s"
            elif token in ['you', 'yourself', 'yourselves']:
                return self._speaker2
            elif token in ['your', 'yours']:
                return self._speaker2 + "'s"
        else:
            if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
                return self._speaker2
            elif token in ['my', 'mine', 'our', 'ours']:
                return self._speaker2 + "'s"
            elif token in ['you', 'yourself', 'yourselves']:
                return self._speaker1
            elif token in ['your', 'yours']:
                return self._speaker1 + "'s"
        return token

    def extract_triples(self, inputs, verbose=True):
        # 0. Assign speakers to ambiguous you/I
        tokens = self._tokenize(inputs)

        # 1. Extract SPO arguments from token sequence
        y_subjs, y_preds, y_objs, subwords = self._argument_module.predict(tokens)
        subjs = bio_tags_to_tokens(subwords, y_subjs.T, one_hot=True)
        preds = bio_tags_to_tokens(subwords, y_preds.T, one_hot=True)
        objs = bio_tags_to_tokens(subwords, y_objs.T, one_hot=True)

        if verbose:
            print('subjs: ', subjs)
            print('preds: ', preds)
            print('objs:  ', objs, '\n')

        # 2. Compute all possible triples
        candidates = [list(triple) for triple in product(subjs, preds, objs)]
        if not candidates:
            return []

        # 3. Score candidate triples
        predictions = self._scoring_module.predict_multi(inputs.split(), candidates)

        # 4. Rank candidates according to entailment predictions
        triples = []
        for y_hat, (subj, pred, obj) in zip(predictions, candidates):
            pol = 'negative' if y_hat[2] > y_hat[1] else 'positive'
            ent = max(y_hat[1], y_hat[2])
            triples.append((ent, (subj, pred, obj, pol)))

        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj, pol)


if __name__ == '__main__':
    model = AlbertTripleExtractor('models/argument_extraction_albert-v2_09_04_2022_multi',
                                  'models/scorer_albert-v2_09_04_2022_multi')
    # Test!
    example = 'i enjoy watching sports but do not want to do homework <eos> what do you like ? <eos> animals , but not cats <eos>'

    print("Input: ", example, '\n')
    for score, triple in model.extract_triples(example):
        print(score, triple)
