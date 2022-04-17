import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from pprint import pprint
from personagpt import PersonaGPT
from persona import Persona
from run_transformer_pipeline import AlbertTripleExtractor


# Suppresses greeting responses
GREETINGS = ['hi', 'hi']


def evaluate(persona, model, num_graphs=10, context_size=8, conf_threshold=0.5):
    for _ in range(num_graphs):
        print('---------------------------------------')
        persona_facts = persona.sample_persona()
        print('\nPERSONA:')
        for item in persona_facts:
            print(item)
        print()

        # Endow chatbot with graph
        bot = PersonaGPT(persona=persona.persona)

        # Store triples extracted from QA-pairs
        outputs = []

        for i in range(len(persona_facts)):
            # Sample question about personal fact
            question = persona.sample_question(i)
            print('USER:', question)

            # Let PersonaGPT respond to last turns
            response = bot.respond(GREETINGS + [question], polarity=persona.get_polarity(i))
            print('BOT: ', response)

            # Extract last three turns
            context_window = ' <eos> '.join([GREETINGS[-1], question, response])

            for conf, triple in model.extract_triples(context_window, verbose=False):
                if conf > conf_threshold:
                    outputs.append(triple)
        print()
        pprint(persona.triples())
        print()
        pprint(outputs)


if __name__ == '__main__':
    model = AlbertTripleExtractor('../../model_transformer/models/2022-04-11')
    persona = Persona('personas.json', num_facts=5)
    evaluate(persona, model, num_graphs=1)
