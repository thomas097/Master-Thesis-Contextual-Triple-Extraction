import sys
sys.path.append('predicate_normalization')

from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from post_processing import PostProcessor
from utils import pronoun_to_speaker_id, speaker_id_to_speaker

from itertools import product
import spacy


class AlbertTripleExtractor:
    def __init__(self, path, base_model='albert-base-v2', sep='<eos>', speaker1='speaker1', speaker2='speaker2'):
        """ Constructor of the Albert-based Triple Extraction Pipeline.

        :param path:       path to savefile
        :param base_model: base model (default: albert-base-v2)
        :param sep:        separator token used to delimit dialogue turns (default: <eos>)
        :param speaker1:   name of user (default: speaker1)
        :param speaker2:   name of system (default: speaker2)
        """
        self._argument_module = ArgumentExtraction(base_model, path=path)
        self._scoring_module = TripleScoring(base_model, path=path)

        self._post_processor = PostProcessor()
        self._nlp = spacy.load('en_core_web_sm')
        self._sep = sep

        # Assign identities to speakers
        self._speaker1 = speaker1
        self._speaker2 = speaker2

    def _tokenize(self, dialog):
        """ Divides up the dialogue into separate turns and dereferences
            personal pronouns 'I' and 'you'.

        :param dialog: separator-delimited dialogue
        :return:       list of tokenized dialogue turns
        """
        # Split dialogue into turns
        turns = [turn.lower().strip() for turn in dialog.split(self._sep)]

        # Tokenize each turn separately (and splitting "n't" off)
        tokens = []
        for turn_id, turn in enumerate(turns):
            # Assign speaker ID to turns (tn=1, tn-1=0, tn-2=1, etc.)
            speaker_id = (len(turns) - turn_id + 1) % 2
            if turn:
                tokens += [pronoun_to_speaker_id(t.lower_, speaker_id) for t in self._nlp(turn)] + ['<eos>']
        return tokens

    def extract_triples(self, dialog, verbose=True, batch_size=32):
        """

        :param dialog:     separator-delimited dialogue
        :param verbose:    whether to print messages (True) or be silent (False) (default: True)
        :param batch_size: If a lot of possible triples exist, batch up processing
        :return:           A list of confidence-triple pairs of the form (confidence, (subj, pred, obj, polarity))
        """
        # Assign unambiguous tokens to you/I
        tokens = self._tokenize(dialog)

        # Extract SPO arguments from token sequence
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
        predictions = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            for y_hat in self._scoring_module.predict(tokens, batch):
                predictions.append(y_hat)

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
            triples.append((ent, (subj, pred, obj, pol)))

        for entailed, (subj, pred, obj, pol) in sorted(triples, key=lambda x: -x[0]):
            yield entailed, (subj, pred, obj, pol)


if __name__ == '__main__':
    model = AlbertTripleExtractor(path='models/2022-04-11')

    # Test!
    example = 'I like photography, but not Spanish lessons<eos> I love that! I learned spanish too. What do you do? <eos> I am an electrical engineer. Not a linguist.'

    for score, triple in model.extract_triples(example):
        print(score, triple)
