from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import random
import re

from templates import *


class PersonaGPT:
    def __init__(self, base_model='af1tang/personaGPT', persona=None):
        self._tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        self._model = AutoModelForCausalLM.from_pretrained(base_model)

        # Set persona
        self._persona = []
        if persona is not None:
            self._persona = self._set_persona(persona)

        # Check for GPU
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model.to(self._device)

    def _set_persona(self, facts):
        return self._tokenizer.encode(''.join(['<|p2|>'] + facts + ['<|sep|>'] + ['<|start|>']))

    def _generate_response(self, input_ids, top_p=.92, max_length=640):
        out = self._model.generate(input_ids, do_sample=True, top_k=10, top_p=top_p,
                                   max_length=max_length, pad_token_id=self._tokenizer.eos_token_id)
        msg = out.cpu().detach().numpy()[0][input_ids.shape[-1]:]
        return list(msg)

    def respond(self, text):
        # Encode user input
        input_ids = self._tokenizer.encode(text + self._tokenizer.eos_token)

        # Generate response from PersonaGPT
        bot_input_ids = torch.LongTensor([self._persona + input_ids]).to(self._device)
        message = self._generate_response(bot_input_ids)

        return self._tokenizer.decode(message, skip_special_tokens=True)


def generate_persona():
    """ Generates a persona from the templates in templates.py
    """
    persona = []
    entity_pool = {}

    for key, options in PERSONA_TEMPLATES.items():
        # Sample template from key category
        template = random.choice(options)

        # Populate template with entities
        entities = re.findall('[A-Z_]+', template)
        for i, slot in enumerate(entities):
            entity = random.choice(ENTITIES[slot])
            template = template.replace(slot, entity, 1)
            entities[i] = (slot, entity)

        persona.append(template)
        entity_pool[key] = entities

    return persona, entity_pool


def generate_question(entity_pool):
    """ Generates questions about the generated persona
    """
    # Pick what topic to ask about
    topic = random.choice(list(entity_pool.keys()))
    entities = entity_pool[topic]

    # Pick a question with said topic
    question = random.choice(PERSONA_QUESTIONS[topic])

    # Populate wildcards if needed
    for i in range(question.count('*')):
        _, entity = entities[i]
        question = question.replace('*', entity, 1)

    # Fix perspective
    question = question.replace('my', 'your')

    return question
