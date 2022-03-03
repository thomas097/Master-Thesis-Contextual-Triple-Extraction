import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import product
import numpy as np
from tqdm import tqdm
from pprint import pprint


class GPTScorer:
    def __init__(self, model='microsoft/DialoGPT-small'):
        print('loading', model)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        print('GPTScorer ready')

    def perplexity(self, strings):
        input_string = ' '.join([s + self._tokenizer.eos_token for s in strings])
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

    def _pronoun_match_score(self, candidate, pronoun):
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

    def similarity(self, pronouns, candidates):
        score = 0.0
        for (cand, _, _), (pron, _) in zip(candidates, pronouns):
            score += self._pronoun_match_score(cand, pron) ** 2
        return score


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

    def _locate_pronouns(self, tokens):
        pronouns = []
        # Add obvious pronouns
        for token in tokens:
            if token.lower_ in self._allowed:
                pronouns.append((token.lower_, token.i))
        # TODO: add ambiguous noun phrases
        return pronouns

    def resolve(self, context, response, beam=4):
        # Tokenize context and response
        context_tokens = self._nlp(context)
        response_tokens = self._nlp(response)

        # Identify unresolved pronouns and possible antecedents
        candidates = self._identify_candidates(context_tokens) #| self._identify_candidates(response_tokens)
        pronouns = self._locate_pronouns(response_tokens)
        print("candidates:", [c for _, c, _ in candidates])
        print('pronouns:', [p for p, _ in pronouns])

        # Score all antecedent-pronoun combinations
        scores = []
        for antecedents in product(candidates, repeat=len(pronouns)):

            # Replace pronouns with antecedents
            alt_response = [t.lower_ for t in response_tokens]
            for i, (_, j) in enumerate(pronouns):
                alt_response[j] = antecedents[i][1]
            alt_response = ' '.join(alt_response)

            # Score alternative string
            perplexity = self._scorer.perplexity([context, alt_response])
            agreement = self._embed.similarity(pronouns, antecedents)
            scores.append((alt_response, agreement / perplexity))

        pprint(scores)

        return max(scores, key=lambda x: x[1])[0]


if __name__ == '__main__':
    coref = Coref('embeddings/glove.10000.300d.txt')
    while True:
        print('---------------------------------------------')
        context = input('context: ')
        response = input('response: ')
        print('\noutput:', coref.resolve(context, response))
