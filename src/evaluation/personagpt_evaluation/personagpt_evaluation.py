from pprint import pprint
from personagpt import PersonaGPT
from persona import Persona
from run_transformer_pipeline import AlbertTripleExtractor


# Helps suppresses greeting responses
GREETINGS = ['hi', 'hi']


def evaluate(persona, model, num_graphs=10, conf_threshold=0.5):
    for _ in range(num_graphs):
        print('---------------------------------------')
        persona_facts = persona.sample_persona()
        print('\nPERSONA:')
        for _, triple in persona_facts:
            print(triple)
        print()

        # Endow chatbot with graph
        bot = PersonaGPT(persona=persona.persona_lines)

        # Store triples extracted from QA-pairs
        outputs = []

        for i in range(len(persona_facts)):
            # Sample question about i-th personal fact
            question = persona.persona_question(i)
            print('USER:', question)

            # Let PersonaGPT respond to last turns
            response = bot.respond(GREETINGS + [question], polarity=persona.persona_polarity(i))
            print('BOT: ', response)

            # Extract last three turns
            context_window = ' <eos> '.join([GREETINGS[-1], question, response])

            for conf, triple in model.extract_triples(context_window, verbose=False):
                if conf > conf_threshold:
                    outputs.append(triple)
        print()
        pprint(outputs)


if __name__ == '__main__':
    model = AlbertTripleExtractor('../../model_transformer/models/2022-04-27')
    persona = Persona('persona_triples.txt', num_facts=5)
    evaluate(persona, model, num_graphs=1)
