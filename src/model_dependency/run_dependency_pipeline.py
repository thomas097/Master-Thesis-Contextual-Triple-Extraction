import spacy
from coreference_resolution import Coref
from pprint import pprint


PRED_LINKS = [('xcomp', 'VERB'), ('aux', 'AUX'), ('prt', 'ADP'), ('prep', 'ADP'),
              ('advmod', 'ADV'), ('aux', 'PART'), ('auxpass', 'AUX')]

ROOT_DEPS = ['ROOT', 'conj', 'advcl', 'ccomp']
OBJ_DEPS = ['dobj', 'pobj', 'attr', 'advmod', 'acomp', 'ccomp']
NP_DEPS = ['det', 'compound']

WH_WORDS = ['how', 'what', 'why', 'who', 'when', 'where']

POS_INTJS = ['yes', 'correct', 'yup', 'yea', 'yeah']
NEG_INTJS = ['no', 'nope', 'nah', 'neh']

PRONOUNS = ['it', 'he', 'his', 'him', 'her', 'she', 'they', 'them', 'their', 'our', 'us', 'we', 'there']


class SpacyTripleExtractor:
    def __init__(self, embeddings_file):
        self._coref = Coref(embeddings_file)
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
        for t in list(root.children) + [root]:
            if t.lower_ in POS_INTJS:
                return 'positive'
            elif t.lower_ in NEG_INTJS:
                return 'negative'
        return 'neutral'

    def _get_triples(self, turn, turn_idx):
        triples = []
        for sent in turn.sents:
            for pred_root in self._get_predicate_roots(sent):
                # SPO
                subject = self._get_subject(pred_root)
                predicate = self._get_predicate(pred_root)
                object_ = self._get_object(predicate)
                polarity = self._get_polarity(pred_root)
                feedback = self._get_feedback_polarity(pred_root)

                # Complete predicate with nested objects (not main object)
                predicate = self._predicate_with_nested_obj(predicate, object_)

                print([subject, predicate, object_, polarity, feedback])

                # Add subject to conjuncts (if needed)
                if not subject:
                    subject = self._get_subject(sent.root)

                # Predicate ellipsis (fails to find object)
                if not subject and not predicate and not object_:
                    object_ = self._get_singular_object(sent.root)

                # Remove Wh-objects, interjections (indicates question)
                if object_ and object_[0].lower_ in WH_WORDS + NEG_INTJS + POS_INTJS:
                    object_ = []

                triples.append([subject, predicate, object_, polarity, feedback, turn_idx])

        return triples

    def _interp_context(self, triples, turns):
        # Co-reference resolution
        for i, (subj, pred, obj, _, _, sent_idx) in enumerate(triples):
            # Resolve subject
            if len(subj) == 1 and subj[0].lower_ in PRONOUNS:
                context = turns[:sent_idx + 1]
                pronoun = subj[0].lower_
                triples[i][0] = self._coref.resolve(context, pronoun).split()

            # Resolve object
            if len(obj) == 1 and obj[0].lower_ in PRONOUNS:
                context = turns[:sent_idx + 1]
                pronoun = obj[0].lower_
                triples[i][2] = self._coref.resolve(context, pronoun).split()

        # If object is missing, inherit object from the response (for question)
        for i, (subj, pred, obj, pol, _, _) in enumerate(triples):
            if i < len(triples) - 1 and subj and pred and not obj:
                next_obj = triples[i + 1][2]
                if next_obj:
                    triples[i][2] = next_obj

        # Predicate ellipsis, e.g. "Do you like cats? no, dogs!"
        for i, (subj, pred, _, _, _, _) in enumerate(triples):
            if not subj and not pred:
                context = [(s, p) for s, p, _, _, _, _ in triples[:i] if s and p]
                if context:
                    new_subj, new_pred = context[-1]
                    triples[i][0] = new_subj
                    triples[i][1] = new_pred

        # Negate previous triples if response is "no"
        for i, (_, _, _, _, feedback, _) in enumerate(triples):
            if i > 0 and feedback == 'negative':
                triples[i - 1][3] = 'negative'

        # Discard everything but SPO and polarity
        triples = [tuple(triple[:4]) for triple in triples]
        return triples

    def extract_triples(self, inputs):
        # Extract triples from each turn in the input
        print('Raw triples:')
        triples = []
        for turn_idx, turn in enumerate(inputs.split('<eos>')):
            triples += self._get_triples(self._nlp(turn.strip()), turn_idx)

        print('Non-contextual:')
        pprint(triples)
        print()

        turns = inputs.split('<eos>')

        # Contextually interpret these triples
        triples = self._interp_context(triples, turns)
        return triples


if __name__ == '__main__':
    parser = SpacyTripleExtractor('embeddings/glove.10000.300d.txt')

    example = 'in my living room watching goodfellas <eos> what is goodfellas , i have never heard of that ? <eos> best movie of all time'
    result = parser.extract_triples(example)
    pprint(result)

    while True:
        print('#' * 20)
        example = input('>> ')
        result = parser.extract_triples(example)

        print('Contextualized:')
        pprint(result)
