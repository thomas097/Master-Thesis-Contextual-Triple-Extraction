from glob import glob
import json
import spacy
import re

# Rules to map different OIE outputs to the same standard
MOVE_TO_OBJ = {'VERB PART VERB': [2],
               'VERB VERB ADP': [1, 2],
               'VERB VERB PART': [1, 2],
               'VERB AUX ADP': [1, 2],
               'VERB PART VERB ADP': [2, 3],
               'INTJ PART VERB ADP': [2, 3],
               'INTJ PART VERB': [2],
               'VERB VERB': [1],
               'ADP VERB': [1],
               'VERB to VERB ADP VERB': [2, 3, 4],
               'VERB to VERB PART VERB': [2, 3, 4],
               'VERB to VERB VERB': [2, 3],
               'feel like VERB ADP': [2, 3],
               'is about VERB': [2],
               'VERB to VERB PRON ADP': [2, 3, 4],
               'VERB VERB ADP ADP': [1, 2, 3],
               'would VERB to VERB': [3],
               'would VERB VERB': [2],
               'VERB to VERB ADV ADP': [2, 3, 4],
               'VERB DET NOUN ADP': [1, 2, 3],
               'VERB to VERB ADP DET NOUN': [2, 3, 4, 5]}

# Simplification rules which strip off auxiliaries from predicates
REMOVE_AUXILIARIES = {"am VERB": [0],
                      "'m VERB": [0],
                      "' m VERB": [0, 1],
                      "'m ADJ ADP": [0],
                      "able to VERB ADP": [0, 1],
                      "AUX ADJ ADP": [0],
                      "are VERB": [0],
                      "is VERB": [0],
                      "do VERB": [0],
                      "'ll VERB": [0],
                      "is VERB ADP": [0],
                      "will VERB ADP": [0],
                      "will VERB PRON": [0],
                      "been VERB ADP": [0]}


class PostProcessor:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')

    def _apply_rule(self, tokens, tags, rules):
        """ Identifies rule to transform token sequence.
        """
        for rule in rules.keys():
            # Unequal length is never a match
            rule_items = rule.split()
            if len(rule_items) != len(tokens):
                continue

            # Check match with rule
            match = True
            for i, item in enumerate(rule_items):
                if item != tokens[i] and item != tags[i]:
                    match = False
                    break

            # Return if match
            if match:
                return rules[rule]
        return []

    @staticmethod
    def _decontract(phrase, is_predicate=False):
        # Decontract of common contractions and fragments
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r" wo ", "will ", ' ' + phrase + ' ')
        phrase = re.sub(r" ca ", "can ", ' ' + phrase + ' ')

        # 's in predicate is always 'to be'
        if is_predicate:
            phrase = re.sub(r"\'s", " is", phrase)

        # Remove double spaces if any
        return re.sub(' +', ' ', phrase.strip())

    def format(self, triple):
        # Fix contractions
        subj = self._decontract(triple[0])
        rel_ = self._decontract(triple[1], is_predicate=True)
        obj_ = self._decontract(triple[2])

        # Get token sequence of arguments
        rel_tags = [t.pos_ for t in self._nlp(rel_)]
        subj = [t.lower_ for t in self._nlp(subj)]
        rel_ = [t.lower_ for t in self._nlp(rel_)]
        obj_ = [t.lower_ for t in self._nlp(obj_)]

        # Are there any auxiliaries that need to go?
        remove_idx = self._apply_rule(rel_, rel_tags, REMOVE_AUXILIARIES)

        # What should be moved to the object
        move_idx = self._apply_rule(rel_, rel_tags, MOVE_TO_OBJ)

        # Apply remove and move rules
        subj2 = ' '.join(subj)
        pred2 = ' '.join([t for i, t in enumerate(rel_) if i not in move_idx + remove_idx])
        obj2_ = ' '.join([t for i, t in enumerate(rel_) if i in move_idx] + obj_)
        return subj2, pred2, obj2_


if __name__ == '__main__':
    pp = PostProcessor()
    print(pp.format(('SPEAKER1 \'s dog', 'likes to \'ve', 'speaker2 \'s bone')))