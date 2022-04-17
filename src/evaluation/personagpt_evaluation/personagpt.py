from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

NEGATION_WORDS = ['not', 'never', 'no', 'nah', "don't", 'nope']


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
        """ Generates a response given the dialogue history (i.e. question about graph)
        """
        # Generate response as sequence of token ids
        out = self._model.generate(inputs=input_ids, do_sample=True, top_k=10, top_p=top_p,
                                   max_length=max_length, pad_token_id=self._tokenizer.eos_token_id)

        msg = out.cpu().detach().numpy()[0][input_ids.shape[-1]:]

        # Decode message into token string
        return self._tokenizer.decode(list(msg), skip_special_tokens=True)

    def respond(self, turns, polarity):
        # Add fake greetings to minimize chance of "Hi" response
        history_ids = self._tokenizer.encode(' '.join([t + self._tokenizer.eos_token for t in turns]))

        # Generate response from PersonaGPT
        bot_input_ids = torch.LongTensor([self._persona + history_ids]).to(self._device)
        msg = self._generate_response(bot_input_ids)

        # If message likely has wrong polarity, generate again
        msg_polarity = 'negative' if any(x in msg for x in NEGATION_WORDS) else 'positive'
        while msg_polarity != polarity:
            msg = self._generate_response(bot_input_ids)
            msg_polarity = 'negative' if any(x in msg for x in NEGATION_WORDS) else 'positive'

        return msg
