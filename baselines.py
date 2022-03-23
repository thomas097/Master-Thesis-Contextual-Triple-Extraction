import os
import subprocess
import re
from pprint import pprint

NEG_TOKENS = ['not', 'n\'t', 'never', '\'t', 'neither']


class ReVerbBaseline:
    def __init__(self, path='reverb-latest.jar', sep='<eos>'):
        self._path = path
        self._sep = sep

    @staticmethod
    def _negation_in_predicate(pred):
        for token in NEG_TOKENS:
            if token in pred:
                pred = pred.replace(token + ' ', '')
                return 'negative', pred
        return 'positive', pred

    def extract_triples(self, dialogue, verbose=True):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture ReVerb output
        out = subprocess.check_output(['java', '-Xmx512m', '-jar', self._path, 'tmp.txt'],
                                      stderr=subprocess.PIPE)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

        # Extract triple and confidence
        triples = []
        for line in lines:
            items = line.split('\t')
            subj, pred, obj = items[-3:]
            conf = float(items[-7])
            polarity, pred = self._negation_in_predicate(pred)
            triples.append((conf, subj, pred, obj, polarity))
        return triples


class OLLIEBaseline:
    def __init__(self, path='ollie-app-latest.jar', sep='<eos>'):
        self._path = path
        self._sep = sep
        self._pattern = r'(\d,\d+): \(([^;]+); ([^;]+); ([^;]+)\)'

    @staticmethod
    def _negation_in_predicate(pred):
        for token in NEG_TOKENS:
            if token in pred:
                pred = pred.replace(token + ' ', '')
                return 'negative', pred
        return 'positive', pred

    def extract_triples(self, dialogue):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture OLLIE output
        out = subprocess.check_output(['java', '-Xmx512m', '-jar', self._path, 'tmp.txt'],
                                      stderr=subprocess.PIPE)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

        # Extract triple and confidence
        triples = []
        for line in lines:
            if re.match(self._pattern, line):
                conf, subj, pred, obj = re.findall(self._pattern, line)[0]
                polar, pred = self._negation_in_predicate(pred)
                conf = float(conf.replace(',', '.'))
                triples.append((conf, subj, pred, obj, polar))
        return triples


class OpenIEBaseline:
    def __init__(self, sep='<eos>'):
        self._sep = sep

    @staticmethod
    def _negation_in_predicate(pred):
        for token in NEG_TOKENS:
            if token in pred:
                pred = pred.replace(token + ' ', '')
                return 'negative', pred
        return 'positive', pred

    def extract_triples(self, dialogue):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture OpenIE output
        wd = os.getcwd()
        os.chdir("stanford-corenlp-latest/stanford-corenlp-4.4.0")
        out = subprocess.check_output('java -mx1g -cp "*" edu.stanford.nlp.naturalli.OpenIE ../../tmp.txt',
                                      stderr=subprocess.PIPE)
        os.chdir(wd)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

        # Extract triple and confidence
        triples = []
        for line in lines:
            conf, subj, pred, obj = line.split('\t')
            polar, pred = self._negation_in_predicate(pred)
            conf = float(conf)
            triples.append((conf, subj, pred, obj, polar))
        return triples


if __name__ == '__main__':
    example = 'I just got a new car <eos> Do you have a new scooter too? <eos> No, not yet'

    reverb = ReVerbBaseline()
    print('ReVerb:')
    pprint(reverb.extract_triples(example))

    ollie = OLLIEBaseline()
    print('\nOLLIE:')
    pprint(ollie.extract_triples(example))

    openie = OpenIEBaseline()
    print('\nStanford OpenIE:')
    pprint(openie.extract_triples(example))
