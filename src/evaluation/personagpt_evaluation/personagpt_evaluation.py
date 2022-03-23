import sys
sys.path.insert(0, '../model_dependency')
sys.path.insert(0, '../model_transformer')

from personagpt import PersonaGPT
from persona_utils import Persona
from run_transformer_pipeline import AlbertTripleExtractor


def evaluate(persona, model):
    persona_facts = persona.sample_persona()
    print('PERSONA:')
    for item in persona_facts:
        print(item)
    print()

    bot = PersonaGPT(persona=persona.persona)
    for i in range(10):
        print('---------------------------------')
        question = persona.sample_question()
        print('USER:', question)

        response = bot.respond(question)
        print('BOT: ', response)

        # Extract last turns
        context_window = '<eos> ' + question + ' <eos> ' + response + ' <eos>'

        print('TRIPLES:')
        for entailed, triple, polarity in model.extract_triples(context_window):
            print(entailed, triple, polarity)
        print()


if __name__ == '__main__':
    model = AlbertTripleExtractor('../model_transformer/models/argument_extraction_albert-v2_17_03_2022',
                                  '../model_transformer/models/scorer_albert-v2_17_03_2022')
    persona = Persona('personas.json')
    evaluate(persona, model)
