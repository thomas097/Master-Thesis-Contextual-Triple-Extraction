import numpy as np
from tqdm import tqdm
from pprint import pprint


def most_similar_pronoun(token, embeds):
    if token not in embeds:
        return 'sorry'

    pronouns = ['it', 'he', 'his', 'him', 'her', 'she', 'they', 'them', 'their', 'our', 'us', 'we']

    scores = []
    for pronoun in pronouns:
        x0 = embeds[token]
        x1 = embeds[pronoun]
        score = x0.dot(x1) / (np.linalg.norm(x0) * np.linalg.norm(x1))
        scores.append((score, pronoun))
    return sorted(scores)[::-1]


if __name__ == '__main__':
    embedding_file = 'embeddings/glove.10000.300d.txt'

    embeds = dict()
    with open(embedding_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            items = line.strip().split(' ')
            embeds[items[0]] = np.array(items[1:]).astype(np.float)

    while True:
        inputs = input('>> ')
        if inputs in embeds:
            pprint(most_similar_pronoun(inputs, embeds))
        else:
            print('sorry')