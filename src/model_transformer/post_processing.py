from glob import glob
import json
import spacy


def load_annotations(path):
    for fname in glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            yield json.load(file)


def get_triples(annotation):
    tokens = annotation['tokens']
    for i, (subj, pred, obj, _, _) in enumerate(annotation['annotations']):
        if subj or pred or obj:
            yield i, format_triple(subj, pred, obj, tokens)


def format_triple(subj, pred, obj, tokens):
    subj = ' '.join([tokens[i][j] for i, j in subj])
    pred = ' '.join([tokens[i][j] for i, j in pred])
    obj = ' '.join([tokens[i][j] for i, j in obj])
    return subj, pred, obj


MOVE_TO_OBJ = {'VERB VERB ADP': [1, 2],
               'VERB VERB PART': [1, 2],
               'VERB AUX ADP': [1, 2],
               'VERB to VERB ADP': [2, 3],
               'VERB to VERB': [2],
               'VERB VERB': [1],
               'ADP VERB': [1],
               'VERB to VERB ADP VERB': [2, 3, 4],
               'VERB to VERB PART VERB': [2, 3, 4],
               'VERB to VERB VERB': [2, 3],
               'feel like VERB ADP': [2, 3],
               'is about VERB': [2],
               'went out': [1],
               'VERB to VERB PRON ADP': [2, 3, 4],
               'VERB VERB ADP ADP': [1, 2, 3],
               'would VERB to VERB': [3],
               'would VERB VERB': [2],
               'VERB to VERB ADV ADP': [2, 3, 4],
               'VERB DET NOUN ADP': [1, 2, 3],
               'VERB to VERB ADP DET NOUN': [2, 3, 4, 5]}

REMOVE_AUXILIARIES = {"am VERB": [0],
                      "'m VERB": [0],
                      "' m VERB": [0, 1],
                      "'m ADJ ADP": [0],
                      "able to VERB ADP": [0, 1],
                      "AUX ADJ ADP": [0],
                      "are VERB": [0],
                      "is VERB": [0],
                      "'ll VERB": [0],
                      "is VERB ADP": [0],
                      "will VERB ADP": [0],
                      "will VERB PRON": [0],
                      "been VERB ADP": [0]}

REPLACE_CONTR = {"'m": 'am',
                 "' m": 'am',
                 "'ll": 'will',
                 "'re": 'are',
                 "'ve": 'have',
                 "ca": 'can'}


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

    def format(self, triple):
        # Get token sequence of arguments
        subj = [t.lower_ for t in self._nlp(triple[0])]
        pred = [t.lower_ for t in self._nlp(triple[1])]
        obj_ = [t.lower_ for t in self._nlp(triple[2])]
        pred_tags = [t.pos_ for t in self._nlp(triple[1])]

        # Are there any auxiliaries that need to go?
        remove_idx = self._apply_rule(pred, pred_tags, REMOVE_AUXILIARIES)

        # What should be moved to the object
        move_idx = self._apply_rule(pred, pred_tags, MOVE_TO_OBJ)

        # Apply remove, move and replace rules
        subj = [REPLACE_CONTR[t] if t in REPLACE_CONTR else t for t in subj]
        pred = [REPLACE_CONTR[t] if t in REPLACE_CONTR else t for t in pred]
        obj_ = [REPLACE_CONTR[t] if t in REPLACE_CONTR else t for t in obj_]

        subj2 = ' '.join(subj)
        pred2 = ' '.join([t for i, t in enumerate(pred) if i not in move_idx + remove_idx])
        obj2_ = ' '.join([t for i, t in enumerate(pred) if i in move_idx] + obj_)
        return subj2, pred2, obj2_


if __name__ == '__main__':
    annotations = load_annotations('annotations')

    post = PostProcessor()

    for i, annotation in enumerate(annotations):
        for j, triple in get_triples(annotation):
            print(triple)
            new_triple = post.correct(triple)
            if new_triple != triple:
                print('NEW:', new_triple, '\n')

        print()
        if i == 100:
            break
