import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from personagpt import PersonaGPT
from persona import Persona
from run_transformer_pipeline import AlbertTripleExtractor


def evaluate(persona, model, num_graphs=10, conf_threshold=0.9):
    for _ in range(num_graphs):
        print('---------------------------------------')
        persona_facts = persona.sample_persona()
        print('\nPERSONA:')
        for item in persona_facts:
            print(item)
        print()

        bot = PersonaGPT(persona=persona.persona)
        for i in range(len(persona_facts)):
            question = persona.sample_question(i)
            print('USER:', question)

            response = bot.respond(question)
            print('BOT: ', response)

            # Extract last turns
            context_window = 'hello <eos> ' + question + ' <eos> ' + response

            print('\nTRIPLES:')
            for conf, triple in model.extract_triples(context_window):
                if conf > conf_threshold:
                    print(conf, triple)

            print('\nGROUND-TRUTH:')
            print(persona.get_triple(i))
            print()


if __name__ == '__main__':
    model = AlbertTripleExtractor('../../model_transformer/models/2022-04-11')
    persona = Persona('personas.json', num_facts=2)
    evaluate(persona, model)
