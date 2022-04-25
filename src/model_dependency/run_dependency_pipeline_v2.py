import spacy
from itertools import product
from pprint import pprint
from copy import copy
import networkx as nx
from post_processing import PostProcessor


VERBS = ['VERB', 'AUX']
PRED_ARCS = [('aux', 'AUX'), ('prep', 'ADP')]
SUBJ_DEPS = ['nsubj', 'nsubjpass', 'expl']
PATH_RULES = ['conj', 'ROOT', 'ROOT prep', 'ROOT xcomp prep']

NEG_WORDS = ['no', 'nope', 'nah', 'neh', 'never', 'not']
WH_WORDS = ['how', 'what', 'why', 'who', 'when', 'where', 'whose', 'that']


class SpacyTripleExtractor:
    def __init__(self, sep='<eos>'):
        self._nlp = spacy.load('en_core_web_sm')
        self._post_processing = PostProcessor()
        self._sep = sep

    # Utils

    def _format_triple(self, capsule):
        subj = ' '.join([t.lower_ for t in capsule['subj']])
        pred = ' '.join([t.lower_ for t in capsule['pred']])
        obj_ = ' '.join([t.lower_ for t in capsule['obj']])
        polar = capsule['polar']
        return self._post_processing.format((subj, pred, obj_)) + (polar,)

    @staticmethod
    def _list_arguments(utt):
        nps = [list(NP) for NP in utt.noun_chunks]  # Noun phrases
        adjs = [list(t.subtree) for t in utt if t.pos_ == 'ADJ' and t.head.pos_ in VERBS]
        return nps + adjs

    @staticmethod
    def _get_head(tokens):
        return min(tokens, key=lambda t: len(list(t.ancestors)))

    # Subjects/Objects

    @staticmethod
    def _get_subjects(args):
        # Identifies subject(s) from a list of candidate arguments
        is_subj = lambda tokens: [t for t in tokens if t.dep_ in SUBJ_DEPS]  # [] = False
        return [arg for arg in args if is_subj(arg)]

    def _get_objects(self, args):
        # Objects = all arguments that are not subjects
        subjs = self._get_subjects(args)
        return [arg for arg in args if arg not in subjs]

    # Predicates

    @staticmethod
    def _pred_from_path(path):
        # Traverse subordinates of tokens in path
        tokens = []
        queue = copy(path)
        while queue:
            item = queue.pop(0)
            tokens.append(item)
            for child in item.children:
                if (child.dep_, child.pos_) in PRED_ARCS:
                    queue.append(child)

        return sorted(set(tokens), key=lambda t: t.i)

    def _get_predicate(self, utt, subj, obj):
        # Create Graph from dependency parse
        edges = []
        for token in utt:
            edges += [(token, child) for child in token.children]
        dep_graph = nx.Graph(edges)

        # Seek shortest path between subject and object
        path = nx.shortest_path(dep_graph,
                                source=self._get_head(subj),
                                target=self._get_head(obj))[1:-1]

        # Check whether labeled path of arcs is valid
        labeled_path = ' '.join([node.dep_ for node in path])
        if labeled_path in PATH_RULES:
            # Return predicate and all its subordinates
            return self._pred_from_path(path)
        return []

    # Polarity

    @staticmethod
    def _get_polarity(pred):
        main_polar, feedback = 'positive', 'neutral'
        for token in pred:
            for child in token.children:
                if child.dep_ == 'neg':
                    main_polar = 'negative'
                elif child.dep_ == 'intj' and child.lower_ in NEG_WORDS:
                    feedback = 'negative'
        return main_polar, feedback

    # Triples

    def _triples_from_turn(self, turn, turn_id):
        triples = []
        for utt in self._nlp(turn).sents:
            # Extract set of all possible subjects and objects
            args = self._list_arguments(utt)

            # Split set into subjects and objects
            subjs = self._get_subjects(args)
            objs = self._get_objects(args)

            # Find predicate between subjs and objs (if there is any)
            for subj, obj in product(subjs, objs):
                pred = self._get_predicate(utt, subj, obj)
                if pred:
                    # Compute polarity
                    polar, feedback = self._get_polarity(pred)

                    triples.append((subj, pred, obj, polar, feedback))

                    # Keep track which arguments were used
                    if subj in args:
                        args.remove(subj)
                    if obj in args:
                        args.remove(obj)

            # If there are objects left -> not all triples
            for obj in copy(args):
                # Try to find a predicate
                path = [t for t in self._get_head(obj).ancestors if t.pos_ in VERBS]
                pred = self._pred_from_path(path)

                # See if this predicate has a subject
                subj = []
                if pred:
                    subj = [t for t in self._get_head(pred).children if t.dep_ in SUBJ_DEPS]
                    if subj:
                        subj = [t for t in subj[0].subtree]

                # Merge into triple with polarity
                polar, feedback = self._get_polarity(pred)
                triples.append((subj, pred, obj, polar, feedback))

                if obj in args:
                    args.remove(obj)

            # Add obj independently when subj and pred are elided
            triples += [([], [], arg, 'positive', 'neutral') for arg in args]

        # Make dict format
        out = []
        for subj, pred, obj, polar, feedback in triples:
            out.append({'subj': subj, 'pred': pred, 'obj': obj, 'polar': polar,
                        'feedback': feedback, 'turn': turn_id})
        return out

    @staticmethod
    def _interpret_with_context(triples):
        # Remove 'wh'-words from object positions (questions)
        for triple in triples:
            if len(triple['obj']) == 1 and triple['obj'][0].lower_ in WH_WORDS:
                triple['obj'] = []

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

        # Discard everything but SPO and polarity
        return triples

    def extract_triples(self, dialogue):
        # Analyze each dialog turn separately
        turns = [t.strip() for t in dialogue.split(self._sep)]

        # Extract (partial) triples from each turn
        triples = []
        for turn_id, turn in enumerate(turns):
            triples += self._triples_from_turn(turn, turn_id)

        # Resolve contextual ambiguities
        triples = self._interpret_with_context(triples)

        return [self._format_triple(triple) for triple in triples]


if __name__ == '__main__':
    examples = ["No , I am not very ill .",
                "I really enjoy going to the university",
                "I like cats a lot",
                "I love cats but not tigers",
                "What does that mean?",
                "What do you have ? <eos> green cats",
                "Yes , I'd like to move to another room .<eos>Is there anything uncomfortable in your room ?<eos>No . The air conditioner in this room doesn't work .",
                "Do you like cats ? <eos> No , I hate cats and I don't like dogs either",
                "yes mostly i made some for others<eos>that's nice . are you taking fashion classes ?<eos>no i'm studying public relations",
                "No , I am not ill . <eos> Then , What's the matter with your child ? <eos> Nothing ."]

    parser = SpacyTripleExtractor()
    for example in examples:
        res = parser.extract_triples(example)
        print(example)
        print(res)
        print()
