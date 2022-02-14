import json


def show(ann):
    turns = ann['tokens']
    triples = ann['triples']

    print('Turns:')
    for i, turn in enumerate(turns):
        prefix = '  '
        if i == len(turns) - 1:
            prefix = 'A:'
        elif i == len(turns) - 2:
            prefix = 'Q:'
        print('  ' + prefix, ' '.join(turn))

    print('\nTriples:')
    for triple in triples:
        triple_text = []
        for arg in triple:
            arg_text = ' '.join([turns[i][j] for i, j in arg])
            triple_text.append(arg_text)
        triple_text = '<' + ', '.join(triple_text) + '>'
        print('  ', triple_text)


if __name__ == '__main__':
    PATH = 'annotations/dailydialog_000012.json'

    with open(PATH, 'r') as file:
        annotation = json.load(file)
    show(annotation)
