import os
import re
import subprocess
import spacy

from post_processing import PostProcessor


PRONOUNS = {"speaker1": ['i', 'me', 'myself', 'we', 'us', 'ourselves'],
            "speaker2": ['you', 'yourself', 'yourselves'],
            "speaker1 's": ["my", 'mine', 'our'],
            "speaker2 's": ['your', 'yours']}


class StanfordOpenIEBaseline:
    def __init__(self, path="stanford-corenlp-latest/stanford-corenlp-4.4.0", spacy_model='en_core_web_sm', speaker1='speaker1', speaker2='speaker2', sep='<eos>'):
        """ Constructor of the Stanford OpenIE baseline.

        :param path:         Path to the 'stanford-corenlp-<VERSION>' directory
        :param spacy_model:  SpaCy model (default: en_core_web_sm)
        :param speaker1:     Name of the user (default: speaker1)
        :param speaker2:     Name of the system (default: speaker2)
        :param sep:          Separator used to delimit dialogue turns (default: <eos>)
        """
        self._post_processor = PostProcessor()
        self._nlp = spacy.load(spacy_model)
        self._speaker1 = speaker1
        self._speaker2 = speaker2
        self._path = path
        self._sep = sep

    def _extract_perspective(self, subj, pred, obj):
        """ Identifies negation of triple using SpaCy (not supported by Stanford OpenIE).

        :param subj: subject of the triple
        :param pred: predicate of the triple, containing potential negation
        :param obj:  object of triple
        :return:
        """
        # Identify tokens marking negation
        doc = self._nlp('{} {} {}'.format(subj, pred, obj))
        negations = [t.lower_ for t in doc if t.dep_ == 'neg']

        if not negations:
            return subj, pred, obj, 'positive'

        # If negated, place negation in perspective
        pred = ' '.join([t.lower_ for t in doc if t.dep_ != 'neg'])
        return subj, pred, obj, 'negative'

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
        # Where to output warnings, messages, etc.
        stderr = subprocess.PIPE if not verbose else None
        triples = []

        # Analyze turns separately
        for turn_id, turn in enumerate(dialogue.split(self._sep)):

            # Change working directory to run java from root
            wd = os.getcwd()
            os.chdir(self._path)

            # Write turn out to file
            with open('tmp.txt', 'w', encoding='utf-8') as file:
                file.write(turn.strip())

            out = subprocess.check_output('java -mx1g -cp "*" edu.stanford.nlp.naturalli.OpenIE tmp.txt -triple.strict false', stderr=stderr)
            if verbose:
                print('output:', out)
            os.chdir(wd)

            # Convert b' string to plain text
            lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

            # Extract triples from output lines
            for line in lines:
                if line.strip():
                    conf, subj, pred, obj = line.split('\t')
                    conf = float(conf)

                    # Determine polarity using SpaCy
                    subj, pred, obj, polar = self._extract_perspective(subj, pred, obj)

                    # Disambiguate You/I
                    subj = self._disambiguate_pronouns(subj, turn_id)
                    pred = self._disambiguate_pronouns(pred, turn_id)
                    obj = self._disambiguate_pronouns(obj, turn_id)

                    # Make sure the output conforms to standard
                    subj, pred, obj = self._post_processor.format((subj, pred, obj))

                    triples.append((conf, (subj, pred, obj, polar)))
        return triples


if __name__ == '__main__':
    baseline = StanfordOpenIEBaseline(speaker1='alice', speaker2='bob')
    print(baseline.extract_triples('My beer is cold <eos> I am very tired', verbose=True))