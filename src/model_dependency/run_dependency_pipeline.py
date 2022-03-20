import spacy
from coreference_resolution import Coref
from pprint import pprint

SUBJ_DEPS = ['nsubj', 'nsubjpass', 'expl']

PRED_LINKS = [('xcomp', 'VERB'), ('aux', 'AUX'), ('prt', 'ADP'), ('prep', 'ADP'),
              ('advmod', 'ADV'), ('aux', 'PART'), ('auxpass', 'AUX'), ('attr', 'NOUN')]

ROOT_POS = ['VERB', 'AUX', 'ADP']
ROOT_DEPS = ['ROOT', 'conj', 'advcl', 'ccomp']

OBJ_DEPS = ['dobj', 'pobj', 'attr', 'advmod', 'acomp', 'ccomp']
NP_DEPS = ['det', 'compound', 'amod']

WH_WORDS = ['how', 'what', 'why', 'who', 'when', 'where', 'whose', 'that']

POS_INTJS = ['yes', 'correct', 'yup', 'yea', 'yeah']
NEG_INTJS = ['no', 'nope', 'nah', 'neh']

PRONOUNS = ['it', 'he', 'his', 'him', 'her', 'she', 'they', 'them', 'their', 'our', 'us', 'we', 'there']


class SpacyTripleExtractor:
    def __init__(self, embeddings_file=None):
        self._coref = Coref(embeddings_file) if embeddings_file is not None else None
        self._nlp = spacy.load("en_core_web_sm")

    ## Subjects

    @staticmethod
    def _get_subject(root):
        subjs = [t for t in root.children if t.dep_ in SUBJ_DEPS]
        if subjs:
            return list(subjs[0].subtree)
        return []

    ## Predicates

    @staticmethod
    def _get_predicate_roots(sent):
        return [t for t in sent if t.dep_ in ROOT_DEPS and t.head == sent.root]  # root.root == root (i.e. max level 2)

    @staticmethod
    def _get_predicate(root):
        # Check if the root token is valid
        if root.pos_ in ROOT_POS:

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

    ## Objects

    def _get_object(self, pred_tokens):
        # Identify main object of sentence (acc to dep precedence)
        for root in pred_tokens[::-1]:  # heuristic: prefer left-edge of predicate
            for dep in OBJ_DEPS:
                deps = [t for t in root.children if t.dep_ == dep]
                if deps:
                    return self._get_NP(deps[0])
        return []

    @staticmethod
    def _get_NP(root):
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

    ## Polarity

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

    def _parse_relative_clause(self, root):
        pred_root = [t for t in root.subtree if t.dep_ == 'relcl'][0]
        subject = self._get_subject(pred_root)
        predicate = self._get_predicate(pred_root)
        object_ = self._get_NP(root)
        polarity = self._get_polarity(pred_root)
        return subject, predicate, object_, polarity

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

                # Parse relative clauses
                relcl = [t for t in pred_root.children if t.dep_ == 'relcl']
                if relcl:
                    subject, predicate, object_, polarity = self._parse_relative_clause(pred_root)

                # Predicate ellipsis (fails to find object)
                if not subject and not predicate and not object_:
                    object_ = self._get_NP(sent.root)

                # Remove Wh-objects, interjections (indicates question)
                if object_ and object_[0].lower_ in WH_WORDS + NEG_INTJS + POS_INTJS:
                    object_ = []

                triple = {'subj': subject, 'pred': predicate, 'obj': object_,
                          'polar': polarity, 'feedback': feedback, 'turn': turn_idx}
                triples.append(triple)

        return triples

    def _interpret_with__context(self, triples):
        # # Co-reference resolution
        # for i, (subj, pred, obj, _, _, sent_idx) in enumerate(triples):
        #     # Resolve subject
        #     if len(subj) == 1 and subj[0].lower_ in PRONOUNS:
        #         context = turns[:sent_idx + 1]
        #         pronoun = subj[0].lower_
        #         triples[i][0] = self._coref.resolve(context, pronoun).split()
        #
        #     # Resolve object
        #     if len(obj) == 1 and obj[0].lower_ in PRONOUNS:
        #         context = turns[:sent_idx + 1]
        #         pronoun = obj[0].lower_
        #         triples[i][2] = self._coref.resolve(context, pronoun).split()

        # If object is missing, inherit object from next turn (questions)
        for triple in triples:
            if triple['subj'] and triple['pred'] and not triple['obj']:
                candidates = [t for t in triples if t['turn'] == (triple['turn'] + 1) and t['obj']]  # triples from response with object
                if candidates:
                    triple['obj'] = candidates[0]['obj']

        # Predicate ellipsis, e.g. "Do you like cats? no, dogs!"
        for triple in triples:
            if not triple['subj'] and not triple['pred'] and triple['obj']:
                candidates = [t for t in triples if t['turn'] == (triple['turn'] - 1) and t['subj'] and t['pred']]  # triples from question
                if candidates:
                    triple['subj'] = candidates[-1]['subj']
                    triple['pred'] = candidates[-1]['pred']
                    triple['polar'] = candidates[-1]['polar']

        # Negate previous triple if response denies
        for triple in triples:
            if triple['feedback'] == 'negative':
                candidates = [t for t in triples if t['turn'] == triple['turn'] - 1]
                if candidates:
                    candidates[-1]['polar'] = 'negative'

        return triples

    @staticmethod
    def _format_triple(triple):
        subj = ' '.join([t.lower_ for t in triple['subj']])
        pred = ' '.join([t.lower_ for t in triple['pred']])
        obj = ' '.join([t.lower_ for t in triple['obj']])
        polar = triple['polar']
        return subj, pred, obj, polar

    def extract_triples(self, inputs):
        # Extract triples from each turn in the input
        print('Raw triples:')
        triples = []
        for turn_idx, turn in enumerate(inputs.split('<eos>')):
            triples += self._get_triples(self._nlp(turn.strip()), turn_idx)

        print('\nPreprocessed (no context):')
        pprint([(t['subj'], t['pred'], t['obj'], t['polar'], t['feedback'], t['turn']) for t in triples])
        print()

        # Contextually interpret these triples
        triples = self._interpret_with__context(triples)

        # Discard everything but SPO and polarity
        triples = set([self._format_triple(t) for t in triples])
        return triples


if __name__ == '__main__':
    parser = SpacyTripleExtractor() #'embeddings/glove.10000.300d.txt'

    #example = "in my living room watching goodfellas <eos> what is goodfellas ? i have never heard of that <eos> best movie of all time"
    #example = "Yes , I'd like to move to another room .<eos>Is there anything uncomfortable in your room ?<eos>No . The air conditioner in this room doesn't work ."
    #example = "Do you like cats ? <eos> No , I hate cats, and dogs I don't adore"
    example = "yes mostly i made some for others<eos>that's nice . are you taking fashion classes ?<eos>no i'm studying public relations"
    #example = "No , I am not ill . <eos> Then , What's the matter with your child ? <eos> Nothing ."
    result = parser.extract_triples(example)

    print('Contextualized:')
    pprint(result)