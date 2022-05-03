# Set up Java PATH (required for Windows)
import os
import re
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules
from post_processing import PostProcessor

os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_331/bin/java.exe"


PRONOUNS = {"speaker1": ['i', 'me', 'myself', 'we', 'us', 'ourselves'],
            "speaker2": ['you', 'yourself', 'yourselves'],
            "speaker1 's": ["my", 'mine', 'our'],
            "speaker2 's": ['your', 'yours']}


class LeolaniBaseline:
    def __init__(self, speaker1='speaker1', speaker2='speaker2', sep='<eos>'):
        """ Constructor of the Leolani Knowledge (Triple) Extraction baseline.

        :param speaker1:     Name of the user (default: speaker1)
        :param speaker2:     Name of the system (default: speaker2)
        :param sep:          Separator used to delimit dialogue turns (default: <eos>)
        """
        self._chat = Chat(speaker1)
        self._post_processor = PostProcessor()
        self._analyzer = CFGAnalyzer()
        self._speaker1 = speaker1
        self._speaker2 = speaker2
        self._sep = sep

    @property
    def name(self):
        return "Leolani"

    def _disambiguate_pronouns(self, arg, turn_id):
        """ Replaces 'you' and 'I' by the corresponding referent (speaker1 or speaker2)

        :param arg:     String representing an argument of a triple
        :param turn_id: Index of the turn in the dialogue (0=speaker1, 1=speaker2, 2=speaker1, etc.)
        :return:        Argument with personal and possessive pronouns replaced
        """
        # Split contractions and punctuation from tokens
        arg = re.findall("[\w\d]+|'\w|[.,!?]", arg.lower())

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

    @staticmethod
    def _triple_from_capsule(capsule):
        """ Extracts the triple from the capsule from the extractor.

        :param capsule: Capsule from the triple extractor
        :return:        Tuple of the form (subj, pred, obj, polar)
        """
        # Extract triple
        subj = capsule['subject']['label']
        pred = capsule['predicate']['label']
        obj_ = capsule['object']['label']

        # If polarity is missing
        polarity = 'positive'
        if not capsule['perspective']:
            return subj, pred, obj_, polarity

        # Determine polarity
        if capsule['perspective']['polarity'] == -1:
            polarity = 'negative'

        return subj, pred, obj_, polarity

    def _get_capsules(self, utterance):
        """ Extracts a triple from the utterance

        :param utterance: plain-text string representing the utterance
        :return:          a capsule
        """
        # Extract capsule from turn
        self._chat.add_utterance(utterance)
        self._analyzer.analyze(self._chat.last_utterance)
        return utterance_to_capsules(self._chat.last_utterance)

    def _format_turn(self, turn):
        """ Formats the input to make sure input can be parsed (cased).

        :param turn: dialogue turn
        :return:     dialogue turn with capitalized words
        """
        # Capitalize first word and I
        turn = turn.capitalize().replace('i ', 'I ').replace("i'", "I'")
        return turn

    def extract_triples(self, dialogue, verbose=False):
        """ Extracts a set of triples from an <eos>-delimited dialogue

        :param dialogue: <eos>-delimited dialogue
        :param verbose:  Whether to print the messages of the system (default: False)
        :return:         A set of tuples in the form of (confidence, (subj, pred, obj, polarity))
        """
        # Separate dialogue into individual sentences
        triples = []
        for turn_id, turn in enumerate(dialogue.split(self._sep)):

            # Strip punctuation and trailing whitespace (throws off parser)
            turn = self._format_turn(turn)

            for capsule in self._get_capsules(turn):
                subj, pred, obj, polar = self._triple_from_capsule(capsule)

                # Disambiguate You/I
                subj = self._disambiguate_pronouns(subj, turn_id)
                pred = self._disambiguate_pronouns(pred, turn_id)
                obj = self._disambiguate_pronouns(obj, turn_id)

                subj, pred, obj = self._post_processor.format((subj, pred, obj))

                if verbose:
                    print('Output:', (subj, pred, obj, polar))

                triples.append((1, (subj, pred, obj, polar)))
        return triples


if __name__ == '__main__':
    baseline = LeolaniBaseline(speaker1='alice', speaker2='bob')
    print(baseline.extract_triples('The national anthem of Canada is Merry Christmas', verbose=True))