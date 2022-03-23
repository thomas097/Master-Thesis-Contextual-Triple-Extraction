from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import random
import re


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
