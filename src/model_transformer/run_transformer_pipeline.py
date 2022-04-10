from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from post_processing import PostProcessor
from predicate_normalization import PredicateNormalizer
from utils import pronoun_to_speaker_id, speaker_id_to_speaker

from itertools import product
import spacy


class AlbertTripleExtractor:
    def __init__(self, path, base_model='albert-base-v2', pred_file='', sep='<eos>', speaker1='Thomas', speaker2='Leolani'):
        self._argument_module = ArgumentExtraction(base_model, path=path)
        self._scoring_module = TripleScoring(base_model, path=path)
        self._pred_normalizer = PredicateNormalizer(pred_file) if pred_file else None

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
                tokens += [pronoun_to_speaker_id(t.lower_, speaker) for t in self._nlp(turn)] + ['<eos>']
        return tokens

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

            # Replace SPEAKER* with speaker
            subj = speaker_id_to_speaker(subj, self._speaker1, self._speaker2)
            pred = speaker_id_to_speaker(pred, self._speaker1, self._speaker2)
            obj = speaker_id_to_speaker(obj, self._speaker1, self._speaker2)

            # Fix mistakes, expand contractions
            subj, pred, obj = self._post_processor.format((subj, pred, obj))

            # Normalize predicate (optional)
            if self._pred_normalizer:
                pred, _ = self._pred_normalizer.normalize(subj, pred, obj)

            triples.append((ent, (subj, pred, obj, pol)))

        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj, pol)


if __name__ == '__main__':
    model = AlbertTripleExtractor(path='models/2022-04-10',
                                  pred_file='resources/canonical_exemplars.txt')

    # Test!
    example = 'I love photography ! <eos> What do you do ? <eos> I am a janitor . <eos> My wife is a doctor . <eos>'

    print("Input: ", example, '\n')
    for score, triple in model.extract_triples(example):
        print(score, triple)
