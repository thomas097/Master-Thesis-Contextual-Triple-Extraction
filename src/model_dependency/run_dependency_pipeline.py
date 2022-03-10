import spacy
from pprint import pprint


PRED_LINKS = [('xcomp', 'VERB'), ('aux', 'AUX'), ('prt', 'ADP'), ('prep', 'ADP'),
              ('advmod', 'ADV'), ('aux', 'PART'), ('auxpass', 'AUX')]

ROOT_DEPS = ['ROOT', 'conj', 'advcl', 'ccomp']
OBJ_DEPS = ['dobj', 'pobj', 'attr', 'advmod', 'acomp', 'ccomp']
NP_DEPS = ['det', 'compound']

POS_INTJS = ['yes', 'correct', 'yup', 'yea', 'yeah']
NEG_INTJS = ['no', 'nope', 'nah', 'neh']


class SpacyTripleExtractor:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _get_predicate_roots(sent):
        return [t for t in sent if t.dep_ in ROOT_DEPS and t.head == sent.root]  # root.root == root

    @staticmethod
    def _get_subject(root):
        subjs = [t for t in root.children if 'subj' in t.dep_]
        if subjs:
            return list(subjs[0].subtree)
        return []

    @staticmethod
    def _get_predicate(root):
        # Check if the root token is valid
        if root.pos_ in ['VERB', 'AUX']:

            # Depth first traversal to create predicate
            queue = [root]
            tokens = [root]
            while queue:
                token = queue.pop(0)
                for child in token.children:
                    if (child.dep_, child.pos_) in PRED_LINKS:
                        tokens.append(child)
                        queue.append(child)

            # Ensure sentence-order
            return sorted(tokens, key=lambda t: t.i)
        return []

    @staticmethod
    def _predicate_with_nested_obj(pred, obj):
        # Add left over objects back to predicate if not main object
        new_pred = pred
        for pred_token in pred:
            for child in pred_token.children:
                if child not in obj and 'obj' in child.dep_:
                    new_pred += list(child.subtree)
        return sorted(new_pred, key=lambda t: t.i)

    @staticmethod
    def _get_object(pred_tokens):
        # Identify main object of sentence (acc to dep precedence)
        for root in pred_tokens[::-1]:  # heuristic: prefer left-edge of predicate
            for dep in OBJ_DEPS:
                deps = [t for t in root.children if t.dep_ == dep]
                if deps:
                    return list(deps[0].subtree)
        return []

    @staticmethod
    def _get_singular_object(root):
        # Get object when there is predicate ellipsis
        queue = [root]
        tokens = [root]
        while queue:
            token = queue.pop(0)
            for child in token.children:
                if child.dep_ in NP_DEPS:
                    queue.append(child)
                    tokens.append(child)
        return sorted(tokens, key=lambda t: t.i)

    @staticmethod
    def _get_polarity(root):
        # Check for negation of local root
        negations = [t for t in root.children if t.dep_ == 'neg']
        if negations:
            return 'negative'
        return 'positive'

    @staticmethod
    def _get_feedback_polarity(root):
        # Check for polar interjections
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
                # SPO
                subject = self._get_subject(pred_root)
                predicate = self._get_predicate(pred_root)
                object_ = self._get_object(predicate)
                polarity = self._get_polarity(pred_root)
                feedback = self._get_feedback_polarity(pred_root)

                #spacy.displacy.serve(sent) # for debugging

                # Complete predicate with nested objects (not main object)
                predicate = self._predicate_with_nested_obj(predicate, object_)

                # Add subject to conjuncts (if needed)
                if not subject:
                    subject = self._get_subject(sent.root)

                # Predicate ellipsis (fails to find object)
                if not subject and not predicate and not object_:
                    object_ = self._get_singular_object(sent.root)

                triples.append([subject, predicate, object_, polarity, feedback])

        return triples

    def _interp_context(self, triples):
        # Inherent predicate for object-only triples, e.g. "Do you like cats? dogs"!
        for i, (subj, pred, obj, pol, feedback) in enumerate(triples):
            if not subj and not pred:
                context = [(s, p) for s, p, _, _, _ in triples[:i] if s and p]  # subject and predicate are there
                if context:
                    prev_subj, prev_pred = context[-1]
                    triples[i] = [prev_subj, prev_pred, obj, pol, feedback]

        # Yes/no feedback
        for i, triple in enumerate(triples):
            if i > 0 and triple[4] == 'negative':
                triples[i - 1][3] = 'negative'

        # Remove everything but SPO and polarity
        triples = [tuple(triple[:4]) for triple in triples]
        return triples

    def extract_triples(self, inputs):
        # Extract triples from each turn in the input
        triples = []
        for turn in inputs.split('<eos>'):
            triples += self._get_triples(self._nlp(turn.strip()))

        # Contextually interpret these triples
        triples = self._interp_context(triples)
        return triples


if __name__ == '__main__':
    parser = SpacyTripleExtractor()
    example = 'not really , my roomies all love animals though . all three of them<eos>do you have any trouble losing weight ?<eos>i do not think so lol . i tend to stay in shape'
    result = parser.extract_triples(example)
    pprint(result)