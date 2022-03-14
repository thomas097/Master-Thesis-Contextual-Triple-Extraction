import sys
sys.path.insert(0, '../model_dependency')
sys.path.insert(0, '../model_transformer')

from personagpt import *
from run_transformer_pipeline import AlbertTripleExtractor


def evaluate(model):
    persona, entity_pool = generate_persona()
    print('PERSONA:')
    for item in persona:
        print(item)
    print()

    bot = PersonaGPT(persona=persona)
    for i in range(10):
        print('---------------------------------')
        question = generate_question(entity_pool)
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
    model = AlbertTripleExtractor('../model_transformer/models/argument_extraction_albert-v2_14_03_2022',
                                  '../model_transformer/models/scorer_albert-v2_14_03_2022')
    evaluate(model)
