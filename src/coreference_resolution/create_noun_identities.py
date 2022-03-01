import spacy
from collections import defaultdict
from tqdm import tqdm
import json


def load_dailydialogs(path, batch_size=80):
    batch = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            batch.append(line.strip().replace('__eou__', ''))
            if len(batch) > batch_size:
                yield batch
                batch = []


def load_personachat(path, batch_size=80):
    batch = []
    with open(path, 'r', encoding='utf-8') as file:
        for part in ['train', 'valid']:
            data = json.load(file)[part]
            print(len(data))

            # Loop through utterances in dialogs
            for item in data:
                history = item['utterances'][-1]['history'][:5]
                dialog = ' '.join(history).replace('__ SILENCE __', '')
                batch.append(dialog)

                if len(batch) > batch_size:
                    yield batch
                    batch = []


def load_cornell_dialogs(lines_path, index_path, batch_size=80):
    index = dict()
    with open(lines_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            id_, _, _, _, utt = line.split(' +++$+++ ')
            index[id_] = utt.strip()

    batch = []
    with open(index_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            convo = eval(line.strip().split(' +++$+++ ')[-1])
            convo = ' '.join([index[l] for l in convo])
            batch.append(convo)
            if len(batch) > batch_size:
                yield batch
                batch = []


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    WINDOW = 10
    MIN_COUNT = 25
    DISABLE = ['ner', 'entity_linker', 'entity_ruler', 'textcat', 'textcat_multilabel',
               'lemmatizer', 'morphologizer', 'transformer']
    ALLOWED_PRONOUNS = ['']

    readers = [load_personachat('resources/personachat_self_original.json'),
               load_cornell_dialogs('resources/movie_lines.txt', 'resources/movie_conversations.txt'),
               load_dailydialogs('resources/dialogues_text.txt')]

    counter = defaultdict(lambda: defaultdict(int))
    for reader in readers:
        for batch in tqdm(reader):
            for tokens in nlp.pipe(batch, disable=DISABLE):

                # Identify pronouns and noun phrases (possible antecedents)
                PRONs = []
                NPs = []
                for token in tokens:
                    if token.pos_ == 'NOUN':
                        NP = ' '.join([t.lower_ for t in token.subtree])
                        NPs.append((NP, token.i))
                        NPs.append((token.lower_, token.i))
                    if token.pos_ == 'PRON' and token.lower in ALLOWED_PRONOUNS:
                        PRONs.append((token.lower_, token.i))

                # Check if in window
                for pron, i in PRONs:
                    for NP, j in NPs:
                        if abs(i - j) < WINDOW and j < i:  # antecedent before pronoun
                            counter[NP][pron] += 1

    # Convert to gender probabilities
    counter = {NP: {x: y / sum(v.values()) for x, y in v.items()} for NP, v in counter.items() if sum(v.values()) > MIN_COUNT}

    # Save to file
    with open('noun_identities.json', 'w') as file:
        json.dump(counter, file)
