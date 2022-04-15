from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch


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

    def _generate_response(self, input_ids, polarity='positive', top_p=.92, max_length=640):
        # Incentivize negative words when polarity is negative
        negation_words = ['not', 'never', 'no', 'nah', "don't", 'nope']
        negation_ids = [self._tokenizer.encode(w, add_special_tokens=False, add_prefix_space=True) for w in negation_words]

        if polarity == 'negative':
            out = self._model.generate(inputs=input_ids, do_sample=True, top_k=10, top_p=top_p,
                                       max_length=max_length, pad_token_id=self._tokenizer.eos_token_id,
                                       force_words_ids=[negation_ids])  # Incentivize negative polarity
        else:
            out = self._model.generate(inputs=input_ids, do_sample=True, top_k=10, top_p=top_p,
                                       max_length=max_length, pad_token_id=self._tokenizer.eos_token_id,
                                       bad_words_ids=negation_ids)  # Generate no negative words!

        msg = out.cpu().detach().numpy()[0][input_ids.shape[-1]:]
        return list(msg)

    def respond(self, turns, polarity):
        # Add fake greetings to minimize chance of "Hi" response
        history_ids = self._tokenizer.encode(' '.join([t + self._tokenizer.eos_token for t in turns]))

        # Generate response from PersonaGPT
        bot_input_ids = torch.LongTensor([self._persona + history_ids]).to(self._device)
        message = self._generate_response(bot_input_ids, polarity)

        return self._tokenizer.decode(message, skip_special_tokens=True)
