import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from personagpt import PersonaGPT
from persona_utils import Persona
from run_transformer_pipeline import AlbertTripleExtractor


def evaluate(persona, model, conf_threshold=0.5):
    persona_facts = persona.sample_persona()
    print('\nPERSONA:')
    for item in persona_facts:
        print(item)
    print()

    bot = PersonaGPT(persona=persona.persona)
    for i in range(len(persona_facts)):
        print('---------------------------------')
        question = persona.sample_question(i)
        print('USER:', question)

        response = bot.respond(question)
        print('BOT: ', response)

        # Extract last turns
        context_window = '<eos> ' + question + ' <eos> ' + response + ' <eos>'

        print('\nTRIPLES:')
        for conf, triple in model.extract_triples(context_window):
            if conf > conf_threshold:
                print(conf, triple)

        print('\nGROUND-TRUTH:')
        print(persona.get_triple(i))
        print()


if __name__ == '__main__':
    model = AlbertTripleExtractor('../../model_transformer/models/argument_extraction_albert-v2_08_04_2022_multi',
                                  '../../model_transformer/models/scorer_albert-v2_06_04_2022_multi')
    persona = Persona('personas.json')
    evaluate(persona, model)
