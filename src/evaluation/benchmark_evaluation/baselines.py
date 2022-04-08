import os
import sys
import subprocess
import re
from pprint import pprint

# Set up Java PATH (required for Windows)
os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_321/bin/java.exe"

# To find Leolani Triple Extractor
sys.path.insert(0, 'cltl-knowledgeextraction-main/src')
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules

# To allow sentence-internal polarity to be inferred
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

    def extract_triples(self, dialogue, verbose=False):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture ReVerb output
        out = subprocess.check_output(['java', '-Xmx512m', '-jar', self._path, 'tmp.txt'],
                                      stderr=subprocess.PIPE)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

        if verbose:
            pprint(lines)

        # Extract triple and confidence
        triples = []
        for line in lines:
            items = line.split('\t')
            if len(items) < 3:
                continue
            subj, pred, obj = items[-3:]
            conf = float(items[-7])
            polarity, pred = self._negation_in_predicate(pred)
            triples.append((conf, (subj, pred, obj, polarity)))
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

    def extract_triples(self, dialogue, verbose=False):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture OLLIE output
        out = subprocess.check_output(['java', '-Xmx512m', '-jar', self._path, 'tmp.txt'],
                                      stderr=subprocess.PIPE)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')

        if verbose:
            pprint(lines)

        # Extract triple and confidence
        triples = []
        for line in lines:
            if re.match(self._pattern, line):
                conf, subj, pred, obj = re.findall(self._pattern, line)[0]
                polar, pred = self._negation_in_predicate(pred)
                conf = float(conf.replace(',', '.'))
                triples.append((conf, (subj, pred, obj, polar)))
        return triples


class StanfordOpenIEBaseline:
    def __init__(self, path="stanford-corenlp-latest/stanford-corenlp-4.4.0", sep='<eos>'):
        self._path = path
        self._sep = sep

    @staticmethod
    def _negation_in_predicate(pred):
        for token in NEG_TOKENS:
            if token in pred:
                pred = pred.replace(token + ' ', '')
                return 'negative', pred
        return 'positive', pred

    def extract_triples(self, dialogue, verbose=False):
        # Write dialogue out to file
        with open('tmp.txt', 'w', encoding='utf-8') as file:
            lines = [l.strip() for l in dialogue.split(self._sep) if l.strip()]
            file.write('\n'.join(lines))

        # Capture OpenIE output
        wd = os.getcwd()
        os.chdir(self._path)
        out = subprocess.check_output('java -mx1g -cp "*" edu.stanford.nlp.naturalli.OpenIE ../../tmp.txt',
                                      stderr=subprocess.PIPE)
        os.chdir(wd)
        lines = out.decode('UTF-8').strip().replace('\r', '').split('\n')
        lines = [line for line in lines if line.strip() != '']

        if verbose:
            pprint(lines)

        # Extract triple and confidence
        triples = []
        for line in lines:
            conf, subj, pred, obj = line.split('\t')
            polar, pred = self._negation_in_predicate(pred)
            conf = float(conf)
            triples.append((conf, (subj, pred, obj, polar)))
        return triples


class LeolaniBaseline:
    def __init__(self, name='I', sep='<eos>'):
        self._chat = Chat(name)
        self._analyzer = CFGAnalyzer()
        self._sep = sep

    @staticmethod
    def _preprocess(string):
        return string.replace('-', ' ').lower()

    def _triple_from_capsule(self, capsule):
        # Extract triple
        subj = self._preprocess(capsule['subject']['label'])
        pred = self._preprocess(capsule['predicate']['label'])
        obj_ = self._preprocess(capsule['object']['label'])

        # In case there is no perspective
        polarity = 'positive'
        if not capsule['perspective']:
            return subj, pred, obj_, polarity

        # Determine polarity
        if capsule['perspective']['polarity'] < 0.5:
            polarity = 'negative'

        return subj, pred, obj_, polarity

    def _get_capsules(self, utterance):
        # Extract capsule from turn
        self._chat.add_utterance(utterance)
        self._analyzer.analyze(self._chat.last_utterance)
        return utterance_to_capsules(self._chat.last_utterance)

    def extract_triples(self, dialogue, verbose=True):
        # Separate dialogue into individual sentences
        triples = []
        for turn in dialogue.split(self._sep):
            capsules = self._get_capsules(turn)
            for capsule in capsules:
                triples.append(self._triple_from_capsule(capsule))
        return triples


if __name__ == '__main__':
    example = 'I will get a new car <eos> Do you have a new scooter too? <eos> No, not yet'

    leolani = LeolaniBaseline()
    print('Leolani:')
    pprint(leolani.extract_triples(example))

    reverb = ReVerbBaseline()
    print('\nReVerb:')
    pprint(reverb.extract_triples(example))

    ollie = OLLIEBaseline()
    print('\nOLLIE:')
    pprint(ollie.extract_triples(example))

    openie = StanfordOpenIEBaseline()
    print('\nStanford OpenIE:')
    pprint(openie.extract_triples(example))
