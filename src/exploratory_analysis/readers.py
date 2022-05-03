import re
import glob
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def read_bnc(path='resources/british_national_corpus'):
    print('Loading BNC')
    dialogue = []
    for fname in glob.glob(path + '/*.xml'):
        with open(fname, 'r', encoding='utf-8') as file:
            for line in file:
                text = ' '.join(re.findall(r'">([\d\w\s\-,.?!]+)</w>', line))
                text = re.sub('\s\s+', ' ', text).lower().strip()  # remove double whitespace, lowercase and strip

                if text:
                    dialogue.append(text)
                elif dialogue:
                    yield dialogue
                    dialogue = []


def read_circa(path='resources/circa/circa-data.tsv'):
    print('Loading Circa')
    df = pd.read_csv(path, sep='\t')
    for _, row in df.iterrows():
        question = row['question-X'].lower().strip()
        answer = row['answer-Y'].lower().strip()
        yield [question, answer]


def read_coqa(path='resources/coqa/coqa-dev-v1.0.json'):
    print('Loading CoQA')
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)['data']
        for item in data:
            questions = [q['input_text'].lower().strip() for q in item['questions']]
            answers = [a['input_text'].lower().strip() for a in item['answers']]

            # Divide turns into even and odd (questions and answers)
            dialogue = []
            for i in range(len(questions) + len(answers)):
                if i % 2 == 0:
                    dialogue.append(questions[i // 2])
                else:
                    dialogue.append(answers[(i - 1) // 2])
            yield dialogue


def read_daily_dialog(path='resources/ijcnlp_dailydialog/dialogues_text.txt'):
    print('Loading Daily Dialog')
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            dialogue = [t.lower().strip() for t in line.split('__eou__')]
            yield dialogue


def read_personachat(path='resources/personachat/personachat_self_original.json'):
    print('Loading Persona-Chat')
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)['train']
        for item in data:
            utterances = [u.lower().strip() for u in item['utterances'][-1]['history']]
            yield utterances


def read_switchboard(path='resources/switchboard'):
    print('Loading Switchboard')
    for fname in glob.glob(path + '/*/*.utt'):
        with open(fname, 'r', encoding='utf-8') as file:

            started = False
            dialogue = ['']
            for line in file:
                if started and line.strip():
                    text = line.split(':')[-1]
                    text = re.sub('[\[\]\{\}+/\(\)]|<[\w_\s]+>', '', text)  # Remove markup
                    dialogue[-1] = dialogue[-1] + ' ' + text

                elif started and dialogue[-1]:
                    dialogue.append('')

                if line.startswith('='):
                    started = True

            yield [re.sub('\s\s+', ' ', t).lower().strip() for t in dialogue[:-1]]


def read_ubuntu(path='resources/ubuntu/Ubuntu-dialogue-corpus/dialogueText.csv', _max=100000):
    print('Loading Ubuntu')
    df = pd.read_csv(path, sep=',')

    # Accumulate responses
    dialogues = defaultdict(list)
    for i, row in tqdm(df.iterrows()):
        id_ = row['dialogueID']
        text = row['text']
        date = row['date']
        dialogues[id_].append((date, text))
        if i > _max:
            break

    # Order responses per dialogue
    for dialogue in dialogues.values():
        if len(dialogue) > 1:
            _, dialogue = zip(*sorted(dialogue, key=lambda x: x[0]))
            yield [str(d).strip().lower() for d in dialogue]


if __name__ == '__main__':
    for x in read_coqa():
        print(x)