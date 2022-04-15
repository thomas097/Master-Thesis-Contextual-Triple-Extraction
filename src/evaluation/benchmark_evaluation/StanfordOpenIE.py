import os
import subprocess


PRONOUNS = [(('i', 'me', 'myself', 'we', 'us', 'ourselves', 'my'), 'speaker1'),
            (("my", 'mine', 'our'), "speaker1 's"),
            (('you', 'yourself', 'yourselves'), 'speaker2'),
            (('your', 'yours'), "speaker2 's")]

NEG_TOKENS = ['not', "n't", 'never', "'t", 'neither']


class StanfordOpenIEBaseline:
    def __init__(self, path="stanford-corenlp-latest/stanford-corenlp-4.4.0", speaker1='speaker1', speaker2='speaker2', sep='<eos>'):
        self._path = path
        self._speaker1 = speaker1
        self._speaker2 = speaker2
        self._sep = sep

    @staticmethod
    def _negation_in_predicate(pred):
        for token in NEG_TOKENS:
            if token in pred:
                pred = pred.replace(token + ' ', '')
                return 'negative', pred
        return 'positive', pred

    @staticmethod
    def _disambuate_pronouns(turn, turn_id):
        turn = ' %s ' % turn.lower()
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
                    turn = turn.replace(' %s ' % pron, ' %s ' % speaker_id)
        return turn

    def extract_triples(self, dialogue, verbose=False):
        # Write dialogue out to file with disambiguated speakers
        lines = []
        for turn_id, turn in enumerate(dialogue.split(self._sep)):
            line = self._disambuate_pronouns(turn.strip(), turn_id).strip()
            lines.append(line)

        with open('tmp.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))

        # Capture OpenIE output from stdout
        wd = os.getcwd()
        os.chdir(self._path)

        if verbose:
            out = subprocess.check_output('java -mx1g -cp "*" edu.stanford.nlp.naturalli.OpenIE ../../tmp.txt')
        else:
            out = subprocess.check_output('java -mx1g -cp "*" edu.stanford.nlp.naturalli.OpenIE ../../tmp.txt',
                                          stderr=subprocess.PIPE)
        os.chdir(wd)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')
        lines = [line for line in lines if line.strip() != '']

        # Extract triple and confidence
        triples = []
        for line in lines:
            conf, subj, pred, obj = line.split('\t')
            polar, pred = self._negation_in_predicate(pred)
            conf = float(conf)
            triples.append((conf, (subj, pred, obj, polar)))
        return triples


if __name__ == '__main__':
    openie = StanfordOpenIEBaseline()
    print(openie.extract_triples('The beer is cold', verbose=True))