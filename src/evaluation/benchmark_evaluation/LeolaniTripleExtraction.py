import os
import sys

# Set up Java PATH (required for Windows)
os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_321/bin/java.exe"

# To find Leolani Triple Extractor from project folder
sys.path.insert(0, 'cltl-knowledgeextraction-main/src')
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules


class LeolaniBaseline:
    def __init__(self, name='speaker1', sep='<eos>'):
        self._chat = Chat(name)
        self._analyzer = CFGAnalyzer()
        self._sep = sep

    @staticmethod
    def _process_names(string):
        return string.replace('-', ' ').lower()

    def _triple_from_capsule(self, capsule):
        # Extract triple
        subj = self._process_names(capsule['subject']['label'])
        pred = self._process_names(capsule['predicate']['label'])
        obj_ = self._process_names(capsule['object']['label'])

        # If polarity is missing
        polarity = 'positive'
        if not capsule['perspective']:
            return subj, pred, obj_, polarity

        # Determine polarity
        if capsule['perspective']['polarity'] == -1:
            polarity = 'negative'

        return subj, pred, obj_, polarity

    def _get_capsules(self, utterance):
        # Extract capsule from turn
        self._chat.add_utterance(utterance.strip())
        self._analyzer.analyze(self._chat.last_utterance)
        return utterance_to_capsules(self._chat.last_utterance)

    def extract_triples(self, dialogue, verbose=False):
        # Separate dialogue into individual sentences
        triples = []
        for turn in dialogue.split(self._sep):

            for capsule in self._get_capsules(turn):
                triple = self._triple_from_capsule(capsule)

                if verbose:
                    print(triple)

                triples.append((1, triple))
        return triples


if __name__ == '__main__':
    print(LeolaniBaseline().extract_triples('My name is not Thomas'))