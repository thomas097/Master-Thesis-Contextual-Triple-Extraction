from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from post_processing import PostProcessor
import re


PRONOUNS = [(("my", 'mine', 'our'), "speaker1 's"),
            (('i', 'me', 'myself', 'we', 'us', 'ourselves', 'my'), 'speaker1'),
            (('your', 'yours'), "speaker2 's"),
            (('you', 'yourself', 'yourselves'), 'speaker2')]


class NeuralOpenIEBaseline:
    def __init__(self, path='https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz', sep='<eos>'):
        self._predictor = Predictor.from_path(path)
        self._post_processor = PostProcessor()
        self._sep = sep

    @staticmethod
    def _disambuate_pronouns(turn, turn_id):
        # Split contractions and punctuation from tokens
        turn = ' %s ' % ' '.join(re.findall("[\w\d-]+|'\w|[.,!?]", turn.lower()))

        for pronouns, speaker_id in PRONOUNS:
            # Swap speakers for uneven turns
            if turn_id % 2 == 1:
                if 'speaker1' in speaker_id:
                    speaker_id = speaker_id.replace('speaker1', 'speaker2')
                else:
                    speaker_id = speaker_id.replace('speaker2', 'speaker1')

            # Replace pronoun occurrences with speaker_ids
            for pron in pronouns:
                if ' %s ' % pron in turn:
                    turn = turn.replace(' %s ' % pron, ' ' + speaker_id + ' ')
        return turn

    def extract_triples(self, dialogue, verbose=True):
        triples = []
        for turn_id, turn in enumerate(dialogue.split(self._sep)):
            # Disambiguate you and I
            turn = self._disambuate_pronouns(turn, turn_id).strip()

            # Extract triples
            res = self._predictor.predict(sentence=turn.strip())
            for triple in res['verbs']:
                pred = ' '.join(re.findall('\[V: ([\w\d\s]+)\]', triple['description'])).lower()
                ents = re.findall('\[ARG(\d): ([\w\d\s\-\']+)\]', triple['description'])
                polarity = 'negative' if 'NEG:' in triple['description'] else 'positive'

                for i, arg1 in ents:
                    for j, arg2 in ents:
                        if int(i) < int(j):
                            triples.append((1.0, self._post_processor.format((arg1, pred, arg2)) + (polarity,)))
        return triples