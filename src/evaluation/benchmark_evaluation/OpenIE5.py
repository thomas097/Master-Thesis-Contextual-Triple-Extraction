import sys
sys.path.append('../../model_transformer')
from post_processing import PostProcessor

from pyopenie import OpenIE5
import re


""" IMPORTANT: Before using OpenIE5Baseline().extract_triples(), initialize an OpenIE5 server:
    $ java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar --httpPort 8000
"""

PRONOUNS = [(("my", 'mine', 'our'), "speaker1 's"),
            (('i', 'me', 'myself', 'we', 'us', 'ourselves', 'my'), 'speaker1'),
            (('your', 'yours'), "speaker2 's"),
            (('you', 'yourself', 'yourselves'), 'speaker2')]

NEG_TOKENS = ['not', 'no', "n't", 'never', "'t", 'neither']


class OpenIE5Baseline:
    def __init__(self, sep='<eos>'):
        # Set up server
        self._extractor = OpenIE5('http://localhost:8000')
        self._post_processor = PostProcessor()
        self._sep = sep
        print('OpenIE5 ready!')

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

    def extract_triples(self, dialogue, verbose=False):
        triples = []
        for turn_id, turn in enumerate(dialogue.split(self._sep)):
            # Disambiguate you and I
            turn = self._disambuate_pronouns(turn, turn_id).strip()

            # Perform per-utterance analysis
            for utt in turn.split('.'):

                # Skip when input is single word (OpenIE5 crashes)
                if utt.strip().count(' ') < 1:
                    continue

                # Get triples
                res = self._extractor.extract(utt)
                for triple in res:
                    conf = triple['confidence']
                    subj = triple['extraction']['arg1']['text']
                    rel = triple['extraction']['rel']['text']
                    objects = [t['text'] for t in triple['extraction']['arg2s']]
                    polarity = 'negative' if triple['extraction']['negated'] else 'positive'

                    # Strip negation words from relation
                    rel = rel.split()
                    for neg_word in NEG_TOKENS:
                        if neg_word in rel:
                            rel.remove(neg_word)
                    rel = ' '.join(rel).strip()

                    # Handle coordination
                    for obj in objects:
                        if verbose:
                            print(conf, (subj, rel, obj, polarity))

                        triples.append((conf, self._post_processor.format((subj, rel, obj)) + (polarity,)))
        return triples


if __name__ == '__main__':
    openie5 = OpenIE5Baseline()
    print(openie5.extract_triples('hello<eos>Hi, how are you ?<eos>I am doing okay , went to the store .'))