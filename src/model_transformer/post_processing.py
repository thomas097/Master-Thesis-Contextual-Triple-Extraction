import spacy
import re

# Rules to map different OIE outputs to the same standard
PRED_RULES = {'VERB PART VERB': [0, 1],           # "want to go"
              'VERB VERB ADP': [0],               # "likes walking with"
              'VERB VERB PART': [0],              # "likes walking to"
              'VERB AUX ADP': [0],
              'VERB PART VERB ADP': [0, 1],       # "went to walk with"
              'VERB VERB': [0],                   # "like walking"
              'VERB PART VERB PART': [0, 1],      # "like to walk with"
              'VERB ADP VERB PART': [0, 1],       # "feel like going to"
              'VERB PART VERB VERB': [0, 1],
              'VERB ADP VERB VERB': [0, 1],
              'AUX VERB ADP': [0, 1, 2],          # "can give away"
              'AUX ADP': [0, 1],                  # "is about [a movie]"
              'AUX VERB VERB': [0, 1],            # "Must be [having]"
              'VERB ADP VERB ADP': [0, 1],
              'VERB PART VERB PRON ADP': [0, 1]}


# Simplification rules which strip off auxiliaries from predicates (that signal certainty!)
AUXILIARIES = ["am", "'m", "' m", "are", "is", "do", "'ll", "will", "be", "will", "been"]


class PostProcessor:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def _decontract(phrase, is_predicate=False):
        # Decontract of common contractions and fragments
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\' re", " are", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\' d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\' ll", " will", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\' ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\' m", " am", phrase)
        phrase = re.sub(r" wo ", "will ", ' ' + phrase + ' ')
        phrase = re.sub(r" ca ", "can ", ' ' + phrase + ' ')

        # 's in predicate is always 'to be'
        if is_predicate:
            phrase = re.sub(r"\'s", " is", phrase)

        # Remove double spaces if any
        return re.sub(' +', ' ', phrase.strip())

    def _pos_sequence(self, predicate):
        """ Returns the POS sequence of a given predicate string. We add
            a token "I" to ensure correct pos sequence.
        """
        return ' '.join([t.pos_ for t in self._nlp("I " + predicate)][1:])

    def format(self, triple):
        # Fix contractions
        subj = self._decontract(triple[0])
        pred = self._decontract(triple[1], is_predicate=True)
        obj = self._decontract(triple[2])

        # Get token sequence of arguments
        pred_tags = self._pos_sequence(pred)
        obj_tags = self._pos_sequence(obj)
        pred = [t.lower_ for t in self._nlp(pred)]
        obj = [t.lower_ for t in self._nlp(obj)]

        # Remove auxiliaries if there is a following verb or auxiliary
        if len(pred) > 1:
            for aux in AUXILIARIES:
                if pred[0] == aux and pred_tags.split(' ')[1] in ['AUX', 'VERB']:
                    pred = pred[1:]
                    pred_tags = self._pos_sequence(' '.join(pred))

        # If object starts with PART or ADP; move token to predicate
        if obj_tags.split(' ')[0] in ['PART', 'ADP']:
            pred = pred + [obj.pop(0)]
            pred_tags = self._pos_sequence(' '.join(pred))

        # Simplify predicate by moving parts to object
        for rule, pred_idx in PRED_RULES.items():
            if pred_tags.startswith(rule):
                obj = [t for i, t in enumerate(pred) if i not in pred_idx] + obj
                pred = [t for i, t in enumerate(pred) if i in pred_idx]

        return subj, ' '.join(pred), ' '.join(obj)


if __name__ == '__main__':
    pp = PostProcessor()
    print(pp.format(('speaker1', 'can eat', 'pizza')))