import random
import spacy
import language_tool_python
from pprint import pprint


PRON_SWAPS = {"i": "you", "my": "your", "me": "you", "mine": "your", "myself": "yourself",
              "we": "you", "us": "you", "our": "your", "ours": "yours", "ourselves": "yourselves"}

AUXILIARIES = ['do', 'does', 'am', 'is', 'are', 'can', 'could']


class QuestionNLG:
    def __init__(self):
        # Load language corrector for NLG
        self._checker = language_tool_python.LanguageTool('en-US')
        self._nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def _swap_pronouns(string):
        """ Swaps person of pronouns, such as 'I' -> 'you' and 'my' -> 'your'
        """
        string = ' %s ' % string.lower()
        for pron, repl in PRON_SWAPS.items():
            if ' %s ' % pron in string:
                string = string.replace(' %s ' % pron, ' %s ' % repl)
        return string.strip()

    def _correct_errors(self, question):
        """ Corrects grammar mistakes in questions generated
        """
        # Identify mistakes
        errors = []
        for match in self._checker.check(question):
            if match.replacements:
                start = match.offset
                stop = match.errorLength + match.offset
                repl = match.replacements[0]
                orig = question[start:stop]
                errors.append((start, stop, repl, orig))

        # Correct mistakes
        question2 = list(question)
        for start, stop, repl, orig in errors:
            for i in range(len(question)):
                question2[start] = repl
                if start < i < stop:
                    question2[i] = ""

        return ''.join(question2)

    def generate_polar_question(self, subj, pred, obj):
        """ Generates a polar question from triple; e.g. (i, like cats) -> 'do you like cats?'
        """
        # Change person
        subj = self._swap_pronouns(subj)
        pred = self._swap_pronouns(pred)
        obj = self._swap_pronouns(obj)

        # Move auxiliaries forward
        question = None
        for aux in AUXILIARIES:
            if pred.startswith(aux + ' ') and pred.count(' '):
                pred = pred.replace(aux + ' ', '')
                question = aux + " %s %s %s ?" % (subj, pred, obj)

        # Fallback
        if question is None:
            question = "do %s %s %s ?" % (subj, pred, obj)

        return self._correct_errors(question)

    def generate_open_question(self, subj, pred, obj, polarity):
        """ Generates an open question from triple; e.g. (i, like cats) -> 'what do you like?'
        """
        # Change person
        subj = self._swap_pronouns(subj)
        pred = self._swap_pronouns(pred)
        obj = self._swap_pronouns(obj)

        # Move auxiliaries forward
        prefix_aux = 'do'
        for aux in AUXILIARIES:
            if pred.startswith(aux + ' ') and pred.count(' '):
                pred = pred.replace(aux + ' ', '')
                prefix_aux = aux

        # Check if prefix auxiliary is correct inflectional form
        if prefix_aux == 'am':
            prefix_aux = 'are'

        # Inherit verb complement from obj if there is any, e.g. "go to", "walking on"
        obj_pos = [t.pos_ for t in self._nlp(obj)]
        if len(obj_pos) > 2 and obj_pos[0] in 'AUX VERB' and obj_pos[1] in 'ADP PART':
            pred = pred + ' ' + ' '.join(obj.split(' ')[:2])

        # "go [eat]"
        if len(obj_pos) > 1 and obj_pos[0] in 'AUX VERB':
            pred = pred + ' ' + obj.split(' ')[0]

        # Ask correct question depending on polarity
        if polarity == 'positive':
            question = "what %s %s %s ?" % (prefix_aux, subj, pred)
        else:
            # Add contractions to sound more natural
            if prefix_aux in ['do', 'did', 'are', 'were', 'can', 'could', 'would']:
                question = "what %sn't %s %s ?" % (prefix_aux, subj, pred)
            else:
                question = "what %s not %s %s ?" % (prefix_aux, subj, pred)

        # Correct grammar
        return self._correct_errors(question)


class Persona:
    def __init__(self, triple_file, num_facts=1, perc_negated=0.5, ):
        # Parse triple_file
        with open(triple_file, 'r', encoding='utf-8') as file:
            triples = set([eval(line.strip()) for line in file])

        # Divide into positive and negative triples
        self._pos_triples = [(l, t) for l, t in triples if t[3] == 'positive']
        self._neg_triples = [(l, t) for l, t in triples if t[3] == 'negative']

        # Sample an initial persona
        self._num_facts = num_facts
        self._perc_negated = perc_negated
        self._persona = self.sample_persona()

        # Define NLG component for questions
        self._question_nlg = QuestionNLG()

    @property
    def persona_lines(self):
        return [l for l, _ in self._persona]

    def persona_line(self, i):
        """ Returns the persona line from which the triple was extracted.
        """
        return self._persona[i][0]

    def persona_polarity(self, i):
        """ Returns the polarity of the persona line/triple.
        """
        return self._persona[i][1][3]

    def persona_triple(self, i):
        """ Returns the triple with polarity of persona line.
        """
        return self._persona[i][1]

    def persona_question(self, i):
        """ Uses simple rule-based NLG to generate a polar/open question for a given triple.
        """
        # Person from first to second: "I" -> "you"
        subj, pred, obj_, polarity = self._persona[i][1]

        # Open or polar question?
        qtype = random.choice(['open', 'polar'])
        if qtype == 'polar':
            return self._question_nlg.generate_polar_question(subj, pred, obj_)

        return self._question_nlg.generate_open_question(subj, pred, obj_, polarity)

    def sample_persona(self):
        """ Samples N persona lines randomly with corresponding triples. Ensures
            the ratio of positive/negative is perc_negated (see constructor).
        """
        triples = []

        # Sample positive facts
        num_pos = int(self._num_facts * (1 - self._perc_negated))
        if num_pos > 0:
            triples += random.sample(self._pos_triples, num_pos)

        # If facts left, sample negative triples
        num_neg = self._num_facts - num_pos
        if num_neg > 0:
            triples += random.sample(self._neg_triples, num_neg)

        random.shuffle(triples)

        self._persona = triples
        return triples


if __name__ == '__main__':
    p = Persona('persona_triples.txt', num_facts=15)
    for i in range(15):
        print(p.persona_line(i))
        print(p.persona_triple(i))
        print(p.persona_question(i))
        print()