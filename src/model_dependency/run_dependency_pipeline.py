import spacy
from pprint import pprint


PRED_LINKS = [('xcomp', 'VERB'), ('aux', 'AUX'), ('prt', 'ADP'), ('prep', 'ADP'),
              ('advmod', 'ADV'), ('aux', 'PART')]

ROOT_DEPS = ['ROOT', 'conj', 'advcl', 'ccomp']
OBJ_DEPS = ['dobj', 'pobj', 'attr', 'advmod', 'acomp', 'ccomp']

POS_INTJS = ['yes', 'correct', 'yup', 'yea', 'yeah']
NEG_INTJS = ['no', 'nope', 'nah', 'neh']


class SpacyTripleExtractor:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _get_predicate_roots(sent):
        return [t for t in sent if t.dep_ in ROOT_DEPS and t.head == sent.root]

    @staticmethod
    def _get_subject(root):
        subjs = [t for t in root.children if 'subj' in t.dep_]
        if subjs:
            return list(subjs[0].subtree)
        return []

    @staticmethod
    def _get_predicate(root):
        if root.pos_ in ['VERB', 'AUX']:
            queue = [root]
            tokens = [root]
            while queue:
                item = queue.pop(0)
                for child in item.children:
                    if (child.dep_, child.pos_) in PRED_LINKS:
                        tokens.append(child)
                        queue.append(child)

            return sorted(tokens, key=lambda t: t.i)
        return []

    @staticmethod
    def _predicate_with_nested_obj(pred, obj):
        new_pred = pred
        for pred_token in pred:
            for child in pred_token.children:
                if child not in obj and 'obj' in child.dep_:
                    new_pred += list(child.subtree)
        return sorted(new_pred, key=lambda t: t.i)

    @staticmethod
    def _get_object(pred_tokens):
        # direct objects, attrs, advmod have precedence!
        for root in pred_tokens[::-1]:  # prefer last object
            for dep in OBJ_DEPS:
                deps = [t for t in root.children if t.dep_ == dep]
                if deps:
                    return list(deps[0].subtree)
        return []

    @staticmethod
    def _get_polarity(root):
        negations = [t for t in root.children if t.dep_ == 'neg']
        if negations:
            return 'negative'
        return 'positive'

    @staticmethod
    def _get_response_polarity(root):
        intjs = [t for t in root.children if t.dep_ == 'intj']
        if intjs:
            if intjs[0].lower_ in NEG_INTJS:
                return 'negative'
            elif intjs[0].lower_ in POS_INTJS:
                return 'positive'
        return 'neutral'

    def _get_triples(self, turn):
        triples = []
        for sent in turn.sents:
            for pred_root in self._get_predicate_roots(sent):
                subj = self._get_subject(pred_root)
                pred = self._get_predicate(pred_root)
                obj_ = self._get_object(pred)
                pol_ = self._get_polarity(pred_root)

                # Complete predicate when they include pobjs (if exist)
                pred = self._predicate_with_nested_obj(pred, obj_)

                # Add subject to conjuncts (if needed)
                if not subj:
                    subj = self._get_subject(sent.root)

                triples.append((subj, pred, obj_, pol_))

        return triples

    def _contextual_interpretation(self, triples):
        # TODO: object only triples, e.g. "Do you like cats? dogs"!
        # TODO: Yes/no negation
        return triples

    def extract_triples(self, inputs):
        # Extract triples from each turn in the input
        triples = [self._get_triples(self._nlp(turn)) for turn in inputs.split('<eos>')]

        # Contextually interpret these triples
        triples = self._contextual_interpretation(triples)
        return triples


if __name__ == '__main__':
    parser = SpacyTripleExtractor()
    example = 'have ever written about canada ?<eos>no , write about my hair and how i color it'
    parser.extract_triples(example)