import re
import spacy
from post_processing import PostProcessor
from pyopenie import OpenIE5

PRONOUNS = {"speaker1": ['i', 'me', 'myself', 'we', 'us', 'ourselves'],
            "speaker2": ['you', 'yourself', 'yourselves'],
            "speaker1 's": ["my", 'mine', 'our'],
            "speaker2 's": ['your', 'yours']}


class OpenIE5Baseline:
    def __init__(self, spacy_model='en_core_web_sm', speaker1='speaker1', speaker2='speaker2', sep='<eos>'):
        """ Constructor of the OpenIE5 baseline.

        IMPORTANT: Before using OpenIE5Baseline().extract_triples(), initialize an OpenIE5 server:
        >> java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar --httpPort 8000

        :param spacy_model:  SpaCy model (default: en_core_web_sm)
        :param speaker1:     Name of the user (default: speaker1)
        :param speaker2:     Name of the system (default: speaker2)
        :param sep:          Separator used to delimit dialogue turns (default: <eos>)
        """
        # Set up server
        self._nlp = spacy.load(spacy_model)
        self._extractor = OpenIE5('http://localhost:8000')
        self._post_processor = PostProcessor()
        self._speaker1 = speaker1
        self._speaker2 = speaker2
        self._sep = sep
        print('OpenIE5 ready!')

    @property
    def name(self):
        return "OpenIE5"

    def _strip_negation(self, pred):
        """ Strips negation in the predicate using SpaCy (OpenIE5 keeps it in the
        predicate as well as marking it as an attribute).

        :param pred: predicate of the triple, containing potential negation
        :return:     predicate without negation
        """
        negations = [t.lower_ for t in self._nlp(pred) if t.dep_ == 'neg']

        # If negated, place negation in perspective
        pred = ' ' + pred + ' '
        for token in negations:
            pred = pred.replace(' %s ' % token, ' ')
        return pred

    def _disambiguate_pronouns(self, arg, turn_id):
        """ Replaces 'you' and 'I' by the corresponding referent (speaker1 or speaker2)

        :param arg:     String representing an argument of a triple
        :param turn_id: Index of the turn in the dialogue (0=speaker1, 1=speaker2, 2=speaker1, etc.)
        :return:        Argument with personal and possessive pronouns replaced
        """
        # Split contractions and punctuation from tokens
        arg = re.findall("[\w\d-]+|'\w|[.,!?]", arg.lower())

        for speaker_id, pronouns in PRONOUNS.items():
            # Swap speakers for uneven turns
            if turn_id % 2 == 1:
                if 'speaker1' in speaker_id:
                    speaker_id = speaker_id.replace('speaker1', 'speaker2')
                else:
                    speaker_id = speaker_id.replace('speaker2', 'speaker1')

            # Replace by referent name
            speaker_id = speaker_id.replace('speaker1', self._speaker1).replace('speaker2', self._speaker2)

            # Replace pronoun occurrences with speaker_ids
            arg = [t if t not in pronouns else speaker_id for t in arg]

        return ' '.join(arg)

    def extract_triples(self, dialogue, verbose=False):
        """ Extracts a set of triples from an <eos>-delimited dialogue

        :param dialogue: <eos>-delimited dialogue
        :param verbose:  Whether to print the messages of the system (default: False)
        :return:         A set of tuples in the form of (confidence, (subj, pred, obj, polarity))
        """
        triples = []
        for turn_id, turn in enumerate(dialogue.split(self._sep)):

            # Skip when input is single word (OpenIE5 crashes)
            if turn.strip().count(' ') < 1:
                continue

            # Get triples
            res = self._extractor.extract(turn)
            for triple in res:
                conf = triple['confidence']
                subj = triple['extraction']['arg1']['text']
                pred = triple['extraction']['rel']['text']
                objs = [t['text'] for t in triple['extraction']['arg2s']]
                polar = 'negative' if triple['extraction']['negated'] else 'positive'

                # Remove negating token from predicate with SpaCy
                if polar == 'negative':
                    pred = self._strip_negation(pred)

                # Handle coordination
                for obj in objs:
                    # Disambiguate You/I
                    subj = self._disambiguate_pronouns(subj, turn_id)
                    pred = self._disambiguate_pronouns(pred, turn_id)
                    obj = self._disambiguate_pronouns(obj, turn_id)

                    # Make sure the output conforms to standard
                    subj, pred, obj = self._post_processor.format((subj, pred, obj))

                    triples.append((conf, (subj, pred, obj, polar)))

                    if verbose:
                        print(conf, (subj, pred, obj, polar))
        return triples


if __name__ == '__main__':
    baseline = OpenIE5Baseline(speaker1='alice', speaker2='bob')
    print(baseline.extract_triples('My beer is cold <eos> I am not very tired', verbose=True))