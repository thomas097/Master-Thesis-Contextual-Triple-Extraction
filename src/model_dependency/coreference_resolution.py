import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm


class GPTScorer:
    def __init__(self, model='microsoft/DialoGPT-small'):
        print('loading', model)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        print('GPTScorer ready')

    def perplexity(self, tokens):
        input_string = ' '.join([t if t != '<eos>' else self._tokenizer.eos_token for t in tokens])
        input_ids = self._tokenizer.encode(input_string, return_tensors='pt')
        output = self._model(input_ids, labels=input_ids)
        loss = output.loss.detach().numpy()
        return np.exp(loss / len(input_ids[0]))


class EmbeddingSimilarity:
    def __init__(self, embedding_file, pronouns):
        self._pronouns = pronouns
        self._embeds = dict()
        print('loading', embedding_file)
        with open(embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                items = line.strip().split(' ')
                self._embeds[items[0]] = np.array(items[1:]).astype(np.float)

    def similarity(self, candidate, pronoun):
        if candidate not in self._embeds:
            return 0.0

        # Compute similarity between all pronouns and candidate
        scores = dict()
        cvec = self._embeds[candidate]
        for pron in self._pronouns:
            pvec = self._embeds[pron]
            scores[pron] = cvec.dot(pvec) / (np.linalg.norm(cvec) * np.linalg.norm(pvec))

        min_ = min(scores.values())
        max_ = max(scores.values())
        return (scores[pronoun] - min_) / (max_ - min_)


class Coref:
    def __init__(self, embeddings_file):
        self._nlp = spacy.load('en_core_web_sm')
        self._allowed = ['it', 'he', 'his', 'him', 'her', 'she', 'they', 'them', 'their', 'our', 'us', 'we', 'there']

        # Perplexity/agreement scoring
        self._scorer = GPTScorer()
        self._embed = EmbeddingSimilarity(embeddings_file, self._allowed)

    @staticmethod
    def _identify_candidates(tokens):
        candidates = set()
        for token in tokens:
            if token.pos_ in ['NOUN', 'PROPN']:
                NP = ' '.join([t.lower_ for t in token.subtree])
                head = token.lower_
                candidates.add((head, NP.lower(), token.i))    # NP
                candidates.add((head, token.lower_, token.i))  # Head of NP
        return candidates

    def _locate_pronoun(self, tokens, pronoun):
        if pronoun.lower() not in self._allowed:  # ensure pronoun is valid
            return None
        return [t for t in tokens if t.lower_ == pronoun.lower()][-1]

    def resolve(self, turns, pronoun=None):
        # Tokenize context and response
        tokens = self._nlp('<eos>'.join(turns))

        # Identify possible antecedents
        candidates = self._identify_candidates(tokens)
        if not candidates:
            return ''

        # Locate pronoun in response
        pronoun = self._locate_pronoun(tokens, pronoun)
        if pronoun is None:
            return ''

        # print("candidates:", [c for _, c, _ in candidates])
        # print('pronoun:', pronoun)

        # Score all antecedent-pronoun combinations
        scores = []
        for head, antecedent, j in candidates:

            # Replace pronoun with antecedent
            alt_tokens = [t.lower_ for t in tokens]
            alt_tokens[pronoun.i] = antecedent

            # Score alternative string
            perplexity = self._scorer.perplexity(alt_tokens)
            agreement = self._embed.similarity(head, pronoun.lower_)
            scores.append((antecedent, agreement / perplexity))

        #pprint(scores)
        return max(scores, key=lambda x: x[1])[0]


if __name__ == '__main__':
    coref = Coref('embeddings/glove.10000.300d.txt')
    while True:
        print('---------------------------------------------')
        dialog = input('dialog: ')
        pronoun = input('pronoun: ')
        print('\noutput:', coref.resolve([dialog], pronoun))
