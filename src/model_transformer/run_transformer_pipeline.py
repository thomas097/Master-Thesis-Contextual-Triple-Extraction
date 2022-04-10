from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from post_processing import PostProcessor

from itertools import product
from utils import *
import spacy


class AlbertTripleExtractor:
    def __init__(self, path, base_model='albert-base-v2', sep='<eos>', speaker1='Thomas', speaker2='Leolani'):
        self._argument_module = ArgumentExtraction(base_model, path=path)
        self._scoring_module = TripleScoring(base_model, path=path)

        self._post_processor = PostProcessor()
        self._nlp = spacy.load('en_core_web_sm')
        self._sep = sep

        # Assign identities to speakers
        self._speaker1 = speaker1
        self._speaker2 = speaker2

    def _tokenize(self, inputs):
        # Tokenize each turn separately (and splitting "n't" off)
        tokens = []
        for turn_idx, turn in enumerate(inputs.split(self._sep)):
            speaker = turn_idx % 2
            turn = turn.strip().lower()
            if turn:
                tokens += [self._disambiguate_pronouns(t.lower_, speaker) for t in self._nlp(turn)] + ['<eos>']
        return tokens

    @staticmethod
    def _disambiguate_pronouns(token, turn_idx):
        # Even turns -> speaker1
        if turn_idx % 2 == 0:
            if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
                return 'SPEAKER1'
            elif token in ['my', 'mine', 'our', 'ours']:
                return "SPEAKER1's"
            elif token in ['you', 'yourself', 'yourselves']:
                return 'SPEAKER2'
            elif token in ['your', 'yours']:
                return "SPEAKER2's"
        else:
            if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
                return "SPEAKER2"
            elif token in ['my', 'mine', 'our', 'ours']:
                return "SPEAKER2's"
            elif token in ['you', 'yourself', 'yourselves']:
                return 'SPEAKER1'
            elif token in ['your', 'yours']:
                return "SPEAKER1's"
        return token

    def _replace_speakers(self, arg):
        return arg.replace('SPEAKER1', self._speaker1).replace('SPEAKER2', self._speaker2)

    def extract_triples(self, inputs, verbose=True):
        # Assign unambiguous tokens to you/I
        tokens = self._tokenize(inputs)

        # Extract SPO arguments from token sequence
        print('Extracting arguments:')
        subjs, preds, objs = self._argument_module.predict(tokens)

        if verbose:
            print('subjects:   %s' % subjs)
            print('predicates: %s' % preds)
            print('objects:    %s\n' % objs)

        # List all possible combinations of arguments
        candidates = [list(triple) for triple in product(subjs, preds, objs)]
        if not candidates:
            return []

        # Score candidate triples
        predictions = self._scoring_module.predict(tokens, candidates)
        print('Scored candidates:')

        # Rank candidates according to entailment predictions
        triples = []
        for y_hat, (subj, pred, obj) in zip(predictions, candidates):
            pol = 'negative' if y_hat[2] > y_hat[1] else 'positive'
            ent = max(y_hat[1], y_hat[2])
            subj = self._replace_speakers(subj)
            pred = self._replace_speakers(pred)
            obj = self._replace_speakers(obj)
            triple = self._post_processor.format((subj, pred, obj))
            triples.append((ent, triple + (pol,)))

        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj, pol)


if __name__ == '__main__':
    model = AlbertTripleExtractor('models/2022-04-09')

    # Test!
    example = 'i enjoy watching auto sports <eos> what do you like doing today ? <eos> games , but not boardgames . <eos>'

    print("Input: ", example, '\n')
    for score, triple in model.extract_triples(example):
        print(score, triple)
