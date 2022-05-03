import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from readers import *
from metrics import *


def basic_stats(readers, reader_names):
    print("### BASIC STATS ###")
    results = []
    for reader, name in zip(readers, reader_names):
        dialogues = list(reader)

        num_dialogs = len(dialogues)
        avg_len_dialog = np.mean([len(d) for d in dialogues])
        avg_len_turn = np.mean([len(t.split(' ')) for d in dialogues for t in d])

        results.append({'name': name,
                        'num_dialogs': num_dialogs,
                        'avg_len_dialog': avg_len_dialog,
                        'avg_len_turn': avg_len_turn,
                        'total_turns': int(avg_len_dialog * num_dialogs)})

    print(pd.DataFrame(data=results))


def freq_of_phenomenon(readers, reader_names, metric, max_lines=8000):
    nlp = spacy.load('en_core_web_sm')

    result = []
    for reader, name in zip(readers, reader_names):
        print('Collecting %s stats' % name)

        # Extract pronoun counts from at most max_lines
        reader_result = Counter()
        i = 0
        for dialogue in tqdm(reader):
            for turn in dialogue:
                i += 1
                if i < max_lines:
                    for token in metric(nlp(turn)):
                        reader_result[token] += 1

        result.append(reader_result.most_common(10))

    for name, freq in zip(reader_names, result):
        print(name, freq)


if __name__ == '__main__':
    readers = [read_bnc(),
               read_circa(),
               read_coqa(),
               read_daily_dialog(),
               read_personachat(),
               read_switchboard(),
               read_ubuntu()]
    reader_names = ['BNC (Spoken)', 'Circa', 'CoQA (dev)', 'DailyDialog',
                    'PersonaChat (train)', 'Switchboard', 'Ubuntu Corpus']

    #basic_stats(readers, reader_names)
    freq_of_phenomenon(readers, reader_names, metric=token_count)

