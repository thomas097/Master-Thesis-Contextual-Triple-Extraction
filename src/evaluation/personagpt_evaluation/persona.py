import spacy
from tqdm import tqdm
import random
import json

# Set seed
#random.seed(2)


NEGATE_PERSONA = {'PRON VERB NOUN PUNCT': "0 don't 1 2 3",
                  'PRON AUX DET NOUN PUNCT': '0 1 not 2 3 4',
                  'PRON VERB PART VERB PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB DET NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB PART VERB NOUN PUNCT': "0 don't 1 2 3 4 5",
                  'PRON VERB ADJ NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB ADP DET NOUN PUNCT': "0 don't 1 2 3 4 5",
                  'PRON ADJ NOUN AUX NOUN PUNCT': '0 1 2 3 not 4 5',
                  'PRON ADJ NOUN AUX ADJ PUNCT': '0 1 2 3 not 4 5',
                  'PRON AUX DET ADJ NOUN PUNCT': '0 1 not 2 3 4 5',
                  'PRON AUX DET NOUN NOUN PUNCT': '0 1 not 2 3 4 5',
                  'PRON VERB NUM NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB ADP DET NOUN NOUN PUNCT': "0 don't 1 2 3 4 5 6",
                  'PRON AUX NUM NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB VERB PUNCT': "0 don't 1 2 3",
                  'PRON AUX ADJ PUNCT': '0 1 not 2 3',
                  'PRON VERB NOUN NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON VERB ADP NOUN PUNCT': "0 don't 1 2 3 4",
                  'PRON NOUN AUX DET NOUN PUNCT': '0 1 2 not 3 4 5',
                  'PRON VERB DET ADJ NOUN PUNCT': "0 don't 1 2 3 4 5"}

PERSONA_TRIPLES = {'PRON VERB NOUN PUNCT': ([0], [1], [2]),
                   'PRON AUX DET NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB PART VERB PUNCT': ([0], [1, 2], [3]),
                   'PRON VERB DET NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB PART VERB NOUN PUNCT': ([0], [1, 2], [3, 4]),
                   'PRON VERB ADJ NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB ADP DET NOUN PUNCT': ([0], [1, 2], [3, 4]),
                   'PRON ADJ NOUN AUX NOUN PUNCT': ([0, 1, 2], [3], [4]),
                   'PRON ADJ NOUN AUX ADJ PUNCT': ([0, 1, 2], [3], [4]),
                   'PRON AUX DET ADJ NOUN PUNCT': ([0], [1], [2, 3, 4]),
                   'PRON AUX DET NOUN NOUN PUNCT': ([0], [1], [2, 3, 4]),
                   'PRON VERB NUM NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB ADP DET NOUN NOUN PUNCT': ([0], [1, 2], [3, 4, 5]),
                   'PRON AUX NUM NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB VERB PUNCT': ([0], [1], [2]),
                   'PRON AUX ADJ PUNCT': ([0], [1], [2]),
                   'PRON VERB NOUN NOUN PUNCT': ([0], [1], [2, 3]),
                   'PRON VERB ADP NOUN PUNCT': ([0], [1, 2], [3]),
                   'PRON NOUN AUX DET NOUN PUNCT': ([0, 1], [2], [3, 4]),
                   'PRON VERB DET ADJ NOUN PUNCT': ([0], [1], [2, 3, 4])}

PERSONA_QUESTIONS_POS = {'PRON VERB NOUN PUNCT': ['Do you 1 2 ?', 'What do you 1 ?'],
                         'PRON AUX DET NOUN PUNCT': ['Are you 2 3 ?'],
                         'PRON VERB PART VERB PUNCT': ['You 1 2 3 ?', 'Do you 1 2 3 ?', 'What do you 1 ?', 'Do you 3 ?'],
                         'PRON VERB DET NOUN PUNCT': ['Do you 1 2 3 ?', 'You 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB PART VERB NOUN PUNCT': ['Do you 1 2 3 4 ?', 'You 3 4 ?', 'Do you 1 4 ?'],
                         'PRON VERB ADJ NOUN PUNCT': ['Do you 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB ADP DET NOUN PUNCT': ['Where do you 1 ?', 'Where do you 1 2 ?', 'Do you 1 2 3 4 ?'],
                         'PRON ADJ NOUN AUX NOUN PUNCT': ['What is 0 1 2 ?', 'Do you have a 1 2 ?', '3 your 1 2 4 ?', '3 4 your 1 2 ?'],
                         'PRON ADJ NOUN AUX ADJ PUNCT': ['What is 0 1 2 ?', 'Do you have a 1 2 ?', 'Is 4 your 1 2 ?', '3 your 1 2 4 ?', '3 4 your 1 2 ?'],
                         'PRON AUX DET ADJ NOUN PUNCT': ['Are you 2 3 4 ?', 'You are 2 3 4 ?'],
                         'PRON AUX DET NOUN NOUN PUNCT': ['Are you 2 3 4 ?', 'You are 2 3 4 ?'],
                         'PRON VERB NUM NOUN PUNCT': ['Do you 1 2 3 ?', 'Do you 1 3 ?'],
                         'PRON VERB ADP DET NOUN NOUN PUNCT': ['Did you 1 2 3 4 5 ?', 'Do you 1 2 3 4 5 ?', 'What did you 1 2 ?'],
                         'PRON AUX NUM NOUN PUNCT': ['Do you have 2 3 ?', 'Do you have 3 ?', 'How many 3 do you have ?'],
                         'PRON VERB VERB PUNCT': ['Do you 1 2 ?', 'What do you 1 ?'],
                         'PRON AUX ADJ PUNCT': ['Are you 2 ?'],
                         'PRON VERB NOUN NOUN PUNCT': ['Do you 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB ADP NOUN PUNCT': ['What do you 1 ?', 'Do you 1 2 3 ?', 'You 1 2 3 ?'],
                         'PRON NOUN AUX DET NOUN PUNCT': ['Is your 1 3 4 ?', 'What does your 1 do?'],
                         'PRON VERB DET ADJ NOUN PUNCT': ['What do you 1 ?', 'Do you 1 2 3 4 ?', 'You 1 2 3 4 ?']}

PERSONA_QUESTIONS_NEG = {'PRON VERB NOUN PUNCT': ['Do you 1 2 ?', 'What do you 1 ?'],
                         'PRON AUX DET NOUN PUNCT': ['Are you 2 3 ?'],
                         'PRON VERB PART VERB PUNCT': ['You 1 2 3 ?', 'Do you 1 2 3 ?', 'What do you 1 ?', 'Do you 3 ?'],
                         'PRON VERB DET NOUN PUNCT': ['Do you 1 2 3 ?', 'You 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB PART VERB NOUN PUNCT': ['Do you 1 2 3 4 ?', 'You 3 4 ?', 'Do you 1 4 ?'],
                         'PRON VERB ADJ NOUN PUNCT': ['Do you 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB ADP DET NOUN PUNCT': ['Where do you 1 ?', 'Where do you 1 2 ?', 'Do you 1 2 3 4 ?'],
                         'PRON ADJ NOUN AUX NOUN PUNCT': ['3 your 1 2 4 ?', '3 4 your 1 2 ?'],
                         'PRON ADJ NOUN AUX ADJ PUNCT': ['3 your 1 2 4 ?', '3 4 your 1 2 ?'],
                         'PRON AUX DET ADJ NOUN PUNCT': ['Are you 2 3 4 ?', 'You are 2 3 4 ?'],
                         'PRON AUX DET NOUN NOUN PUNCT': ['Are you 2 3 4 ?', 'You are 2 3 4 ?'],
                         'PRON VERB NUM NOUN PUNCT': ['Do you 1 2 3 ?', 'Do you 1 3 ?'],
                         'PRON VERB ADP DET NOUN NOUN PUNCT': ['Did you 1 2 3 4 5 ?', 'Do you 1 2 3 4 5 ?', 'What did you 1 2 ?'],
                         'PRON AUX NUM NOUN PUNCT': ['Do you have 2 3 ?', 'Do you have 3 ?', 'How many 3 do you have ?'],
                         'PRON VERB VERB PUNCT': ['Do you 1 2 ?', 'What do you 1 ?'],
                         'PRON AUX ADJ PUNCT': ['Are you 2 ?'],
                         'PRON VERB NOUN NOUN PUNCT': ['Do you 1 2 3 ?', 'What do you 1 ?'],
                         'PRON VERB ADP NOUN PUNCT': ['What do you 1 ?', 'Do you 1 2 3 ?', 'You 1 2 3 ?'],
                         'PRON NOUN AUX DET NOUN PUNCT': ['Is your 1 3 4 ?', 'What does your 1 do?'],
                         'PRON VERB DET ADJ NOUN PUNCT': ['What do you 1 ?', 'Do you 1 2 3 4 ?', 'You 1 2 3 4 ?']}

pronouns = {'i': 'speaker1', 'me': 'speaker1', 'myself': 'speaker1',
            'ourselves': 'speaker1', 'my': "speaker1 's", 'mine': "speaker1 's", 'our': "speaker1 's",
            'ours': "speaker1 's", 'you': 'speaker2', 'yourself': 'speaker2', 'yourselves': 'speaker2',
            'your': "speaker2 's", 'yours': "speaker2 's"}


def pronoun_to_speaker_id(triple):
    """ Takes a triple and replaces all occurrances of 'you' and 'i'
        with the corresponding speaker.
    """
    new_triple = []
    for argument in triple:
        argument = ' %s ' % argument
        for pronoun, speaker in pronouns.items():
            if pronoun in argument:
                argument = argument.replace(' %s ' % pronoun, ' %s ' % speaker)
        new_triple.append(argument.strip())
    return tuple(new_triple)


class Persona:
    def __init__(self, persona_file='persona.json', neg_prob=0.25, num_facts=1):
        # Load file with possible persona lines
        with open(persona_file, 'r', encoding='utf-8') as file:
            self._persona_options = json.load(file)

        # Set up initial persona
        self._num_facts = num_facts
        self._neg_prob = neg_prob
        self._persona = []
        self.sample_persona()

    @property
    def persona(self):
        return [fact for _, _, fact, _ in self._persona]  # only return facts

    def triples(self):
        # Get for ith persona the argument indices into the persona tokens
        result = []
        for category, fact, fact2, polarity in self._persona:
            subj_idx, pred_idx, obj_idx = PERSONA_TRIPLES[category]

            # Convert indices to subj-pred-obj triple
            tokens = fact.split(' ')
            subj = ' '.join([tokens[i] for i in subj_idx])
            pred = ' '.join([tokens[i] for i in pred_idx])
            obj_ = ' '.join([tokens[i] for i in obj_idx])

            triple = pronoun_to_speaker_id((subj, pred, obj_)) + (polarity,)
            result.append(triple)
        return result

    def sample_persona(self):
        # Sample a number of persona 'types'
        categories = random.sample(self._persona_options.keys(), self._num_facts)

        # For each type, sample an instantiation of a fact
        self._persona = []
        for category in categories:
            fact = random.choice(self._persona_options[category])

            # Fix some grammar
            fact = fact.replace("'ve", 'have').replace("'m", 'am').replace(" m ", ' am ')

            # Negate fact with a probability of p
            if random.random() < self._neg_prob:
                pattern = NEGATE_PERSONA[category].split(' ')
                tokens = fact.split(' ')

                # Insert "not" or "n't"
                fact2 = ' '.join([tokens[int(p)] if p.isnumeric() else p for p in pattern])
                polarity = 'negative'
            else:
                fact2 = fact
                polarity = 'positive'

            self._persona.append((category, fact, fact2, polarity))

        return self.persona

    def get_polarity(self, i):
        return self._persona[i][3]

    def sample_question(self, i):
        # Sample one persona line to ask about
        category, fact, fact2, polarity = self._persona[i]

        # Sample question about persona
        if polarity == 'positive':
            question = random.choice(PERSONA_QUESTIONS_POS[category])
        else:
            question = random.choice(PERSONA_QUESTIONS_NEG[category])

        # Generate a question for persona i
        tokens = fact.split(' ')
        question = ' '.join([t if not t.isnumeric() else tokens[int(t)] for t in question.split(' ')])

        # Swap 'my' for 'your'
        question = ' %s ' % question.lower()
        if ' my ' in question:
            question = question.replace(' my ', ' your ')
        if ' mine ' in question:
            question = question.replace(' mine ', ' yours ')
        if ' i ' in question:
            question = question.replace(' i ', ' you ')
        return question.strip()


def categorize_personas(persona_file, outfile='personas.json'):
    nlp = spacy.load('en_core_web_sm')

    # Limit the personas to those with the specific tag sequence in PERSONA_QUESTIONS
    # to allow us to manually write rewrite rules to ask questions about them.
    personas = {k: [] for k in PERSONA_QUESTIONS_POS.keys()}

    with open(persona_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for parse in tqdm(nlp.pipe(lines)):
            # Extract token and POS sequence
            tags = ' '.join([t.pos_ for t in parse if t.pos_ != 'SPACE']).strip()
            tokens = ' '.join([t.lower_ for t in parse if t.pos_ != 'SPACE']).strip()

            # Assign persona to tag seq category if new
            if tags in personas and tokens not in personas[tags]:
                personas[tags].append(tokens)

    # Write out persona categories to file
    with open(outfile, 'w', encoding='utf-8') as file:
        json.dump(personas, file)


if __name__ == '__main__':
    p = Persona('personas.json', num_facts=1)
    for _ in range(5):
        print(p.sample_persona())
        print(p.sample_question())
        print()

