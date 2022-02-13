import spacy

# Dependency links allowed for subjects, predicates and objects
SUBJ_DEPS = [('nsubj', 'PRON'), ('nsubj', 'NOUN'), ('nsubj', 'DET'),
             ('nsubjpass', 'PRON'), ('csubj', 'AUX'), ('csubj', 'VERB'), ('expl', 'PRON')]

PRED_DEPS = [('aux', 'AUX'), ('aux', 'VERB'), ('aux', 'PART'), ('auxpass', 'AUX'),
             ('xcomp', 'AUX'), ('xcomp', 'VERB'),
             ('acomp', 'VERB'), ('prt', 'ADP')]

OBJ_DEPS = [('dobj', 'PRON'), ('dobj', 'NOUN'), ('dobj', 'NUM'), ('dobj', 'PROPN'), ('dobj', 'ADJ'),
            ('pobj', 'NOUN'), ('pobj', 'PROPN'), ('pobj', 'NUM'), ('pobj', 'ADJ'), ('pobj', 'PRON'),
            ('advmod', 'ADV'), ('advmod', 'NOUN'), ('advmod', 'ADJ'),
            ('xcomp', 'ADJ'),
            ('ccomp', 'VERB'), ('ccomp', 'AUX'),
            ('attr', 'NOUN'), ('attr', 'PRON'),
            ('prep', 'ADP'), ('npadvmod', 'NOUN'), ('acomp', 'ADJ')]

DENY_INTJ = ['no', 'nah', 'nope', 'neh']


class DependencyTripleParser:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')

    def _roots(self, utterance):
        """ Determines the root node(s) of the utterance.
        """
        utterance = " ".join(utterance)
        return [t for t in self._nlp(utterance) if t.dep_ == 'ROOT']

    @staticmethod
    def _subject(root):
        """ Returns the nsubj subtree closest to the root.
        """
        nsubjs = [t for t in root.children if (t.dep_, t.pos_) in SUBJ_DEPS]
        if nsubjs:
            return [t for t in nsubjs[0].subtree]
        return []

    def _predicate(self, root):
        """ Recursively extracts the verbs belonging to the predicate.
        """
        # Base case
        if not list(root.children):
            return [root]

        # Extract left/right children belonging to predicate
        lefts = [t for t in root.lefts if (t.dep_, t.pos_) in PRED_DEPS]
        rights = [t for t in root.rights if (t.dep_, t.pos_) in PRED_DEPS]

        # Return subtrees of children in left-to-right order
        pred = []
        for token in lefts:
            pred += self._predicate(token)

        pred.append(root)

        for token in rights:
            pred += self._predicate(token)

        return pred

    @staticmethod
    def _object(pred_tokens):
        """ Extracts the object closest to the root.
        """
        obj = []
        for token in pred_tokens:
            for child in token.children:
                already_used = child in obj or child in pred_tokens
                obj += list(child.subtree) if (child.dep_, child.pos_) in OBJ_DEPS and not already_used else []
        return sorted(obj, key=lambda token: token.i)

    @staticmethod
    def _polarity(root):
        """ Determines the polarity of the sentence w.r.t. the extracted triple.
        """
        root_negations = [t for t in root.subtree if t.dep_ == 'neg']
        if root_negations:
            return 0  # negative
        return 1  # positive

    @staticmethod
    def _polarity_from_response(roots):
        """ Determines the polarity of a sentence by the use of interjections in the response.
        """
        # Explicit interjection confirming or denying previous statement?
        root_intjs = []
        for root in roots:
            root_intjs += [t for t in root.subtree if t.dep_ == 'intj']  # has a Yes / No?

        if root_intjs:
            return 0 if root_intjs[0].text.lower() in DENY_INTJ else 1  # negative = 0; positive = 1

        # Contains complex confirmation/denial statement
        for root in roots:
            pass  # TODO

        return 1

    def parse(self, dialog):
        """ Extracts triples and perspectives for the last utterance in the dialog.
        """
        # Parse utterances in dialog
        dialog_utts = [self._roots(utt) for utt in dialog]

        triples, perspectives = [], []

        # Extract triple(s) from last part of each utterance
        for utt in dialog_utts:
            root = utt[-1]
            subj = self._subject(root)
            pred = self._predicate(root)
            obj = self._object(pred)
            triple = (subj, pred, obj)
            persp = self._polarity(root)

            triples.append(triple)
            perspectives.append(persp)

        # If there were previous utterances, set polarity of their triples according to their responses
        if len(dialog_utts) > 1:
            perspectives[-2] = self._polarity_from_response(dialog_utts[-1])

        # Convert triples to list of lists of indices
        index_triples = []
        for i, triple in enumerate(triples):
            idx_triple = [[(i, token.i) for token in arg] for arg in triple]
            index_triples.append(idx_triple)

        return index_triples, perspectives
