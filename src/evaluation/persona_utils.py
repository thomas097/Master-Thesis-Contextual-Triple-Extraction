#import spacy
from tqdm import tqdm
import random
import json

NEGATE_PERSONA = {'PRON VERB NOUN PUNCT': '0 do not 1 2 3',
                  'PRON AUX DET NOUN PUNCT': '0 1 not 2 3 4',
                  'PRON VERB PART VERB PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB DET NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB PART VERB NOUN PUNCT': '0 do not 1 2 3 4 5',
                  'PRON VERB ADJ NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB ADP DET NOUN PUNCT': '0 do not 1 2 3 4 5',
                  'PRON ADJ NOUN AUX NOUN PUNCT': '0 1 2 3 not 4 5',
                  'PRON ADJ NOUN AUX ADJ PUNCT': '0 1 2 3 not 4 5',
                  'PRON AUX DET ADJ NOUN PUNCT': '0 1 not 2 3 4 5',
                  'PRON AUX DET NOUN NOUN PUNCT': '0 1 not 2 3 4 5',
                  'PRON VERB NUM NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB ADP DET NOUN NOUN PUNCT': '0 do not 1 2 3 4 5 6',
                  'PRON AUX NUM NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB VERB PUNCT': '0 do not 1 2 3',
                  'PRON AUX ADJ PUNCT': '0 1 not 2 3',
                  'PRON VERB NOUN NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON VERB ADP NOUN PUNCT': '0 do not 1 2 3 4',
                  'PRON NOUN AUX DET NOUN PUNCT': '0 1 2 not 3 4 5',
                  'PRON VERB DET ADJ NOUN PUNCT': '0 do not 1 2 3 4 5'}

PERSONA_QUESTIONS = {'PRON VERB NOUN PUNCT': ['Do you 1 2 ?',
                                              'What do you 1 ?'],
                     'PRON AUX DET NOUN PUNCT': ['Are you 2 3 ?',
                                                 'What are you ?'],
                     'PRON VERB PART VERB PUNCT': ['You 1 2 3 ?',
                                                   'Do you 1 2 3 ?',
                                                   'What do you 1 ?',
                                                   'Do you 3 ?'],
                     'PRON VERB DET NOUN PUNCT': ['Do you 1 2 3 ?',
                                                  'You 1 2 3 ?',
                                                  'What do you 1 ?'],
                     'PRON VERB PART VERB NOUN PUNCT': ['Do you 1 2 3 4 ?',
                                                        'Do you 3 4 ?',
                                                        'You 3 4 ?',
                                                        'Do you 1 4 ?'],
                     'PRON VERB ADJ NOUN PUNCT': ['Do you 1 2 3 ?',
                                                  'What do you 1 ?'],
                     'PRON VERB ADP DET NOUN PUNCT': ['Where do you 1 ?',
                                                      'Where do you 1 2 ?',
                                                      'Do you 1 2 3 4 ?'],
                     'PRON ADJ NOUN AUX NOUN PUNCT': ['What is 0 1 2 ?',
                                                      'Do you have a 1 2 ?',
                                                      'Is 4 your 1 2 ?'],
                     'PRON ADJ NOUN AUX ADJ PUNCT': ['What is 0 1 2 ?',
                                                     'Do you have a 1 2 ?',
                                                     'Is 4 your 1 2 ?'],
                     'PRON AUX DET ADJ NOUN PUNCT': ['Are you 2 3 4 ?',
                                                     'You are 2 3 4 ?'],
                     'PRON AUX DET NOUN NOUN PUNCT': ['Are you 2 3 4 ?',
                                                      'You are 2 3 4 ?',
                                                      'What are you ?'],
                     'PRON VERB NUM NOUN PUNCT': ['Do you 1 2 3 ?',
                                                  'What do you 1 ?',
                                                  'Do you 1 3 ?'],
                     'PRON VERB ADP DET NOUN NOUN PUNCT': ['Did you 1 2 3 4 5 ?',
                                                           'Do you 1 2 3 4 5 ?',
                                                           'What did you 1 2 ?'],
                     'PRON AUX NUM NOUN PUNCT': ['Do you have 2 3 ?',
                                                 'Do you have 3 ?',
                                                 'How many 3 do you have ?'],
                     'PRON VERB VERB PUNCT': ['Do you 1 2 ?',
                                              'What do you 1 ?'],
                     'PRON AUX ADJ PUNCT': ['What are you ?',
                                            'Are you 2 ?'],
                     'PRON VERB NOUN NOUN PUNCT': ['Do you 1 2 3 ?',
                                                   'What do you 1 ?'],
                     'PRON VERB ADP NOUN PUNCT': ['What do you 1 ?',
                                                  'Do you 1 2 3 ?',
                                                  'You 1 2 3 ?'],
                     'PRON NOUN AUX DET NOUN PUNCT': ['Is your 1 3 4 ?',
                                                      'What does your 1 do?'],
                     'PRON VERB DET ADJ NOUN PUNCT': ['What do you 1 ?',
                                                      'Do you 1 2 3 4 ?',
                                                      'You 1 2 3 4 ?']}


class Persona:
    def __init__(self, persona_file='persona.json', neg_prob=0.3, num_facts=10):
        # Load file
        with open(persona_file, 'r', encoding='utf-8') as file:
            self._persona_options = json.load(file)

        # Set up initial persona
        self._num_facts = num_facts
        self._neg_prob = neg_prob
        self._persona = []
        self.sample_persona()

    @property
    def persona(self):
        return [fact for _, _, fact in self._persona]  # only return (negated) facts

    def sample_persona(self):
        # Sample a number of persona 'types'
        categories = random.sample(self._persona_options.keys(), self._num_facts)

        # For each type, sample an instantiation of a fact
        self._persona = []
        for category in categories:
            simple_fact = random.choice(self._persona_options[category])

            # Fix some grammar
            simple_fact = simple_fact.replace("'ve", 'have').replace("'m", 'am').replace(" m ", ' am ')

            # Randomly negate simple fact
            if random.random() < self._neg_prob:
                pattern = NEGATE_PERSONA[category].split(' ')
                tokens = simple_fact.split(' ')
                final_fact = ' '.join([tokens[int(p)] if p.isnumeric() else p for p in pattern])
            else:
                # Not negated
                final_fact = simple_fact

            self._persona.append((category, simple_fact, final_fact))

        return self.persona

    def sample_question(self):
        # Sample one persona line to ask about
        category, simple_fact, fact = random.choice(self._persona)

        # Sample question about persona
        question = random.choice(PERSONA_QUESTIONS[category])

        # Replace wildcards
        tokens = simple_fact.split(' ')
        question = ' '.join([t if not t.isnumeric() else tokens[int(t)] for t in question.split(' ')])
        return question


def categorize_personas(persona_file, outfile='personas.json'):
    nlp = spacy.load('en_core_web_sm')

    # Limit the personas to those with the specific tag sequence in PERSONA_QUESTIONS
    # to allow us to manually write rewrite rules to ask questions about them.
    personas = {k: [] for k in PERSONA_QUESTIONS.keys()}

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
    p = Persona('personas.json', num_facts=10)
    for _ in range(3):
        print(p.sample_persona())
        for _ in range(5):
            print(p.sample_question())
        print()

