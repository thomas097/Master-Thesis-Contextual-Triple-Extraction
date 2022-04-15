import sys
sys.path.insert(0, '../../model_dependency')
sys.path.insert(0, '../../model_transformer')

from personagpt import PersonaGPT
from persona import Persona
from run_transformer_pipeline import AlbertTripleExtractor


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

        # Init history with greeting to prevent meaningless introduction after <|start|>
        history = ['hi', 'hi']

        # Store triples extracted from pairs
        outputs = []

        for i in range(len(persona_facts)):
            fact = persona_facts[i]
            print('FACT:', fact)

            # Sample question about personal fact
            question = persona.sample_question(i)
            history.append(question)
            print('USER:', question)

            # Let PersonaGPT respond to last turns
            response = bot.respond(history[-context_size:], polarity=persona.get_polarity(i))
            history.append(response)
            print('BOT: ', response)

            # Extract last three turns
            context_window = '<eos>'.join([''] + history[-2:])

            print('\nTRIPLES:')
            triples = []
            for conf, triple in model.extract_triples(context_window):
                if conf > conf_threshold:
                    triples.append(triple)
                    print(conf, triple)

            print('\nGROUND-TRUTH:')
            ground_truth = persona.get_triple(i)
            print(ground_truth)
            print()

            # Save to file to analyze
            outputs.append((ground_truth, question, response) + tuple(triples))
            with open('evaluation_results.txt', 'w', encoding='utf-8') as file:
                for output in outputs:
                    for item in output:
                        file.write(str(item) + '\n')
                    file.write('\n')


if __name__ == '__main__':
    model = AlbertTripleExtractor('../../model_transformer/models/2022-04-11')
    persona = Persona('personas.json', num_facts=5)
    evaluate(persona, model, num_graphs=1)
