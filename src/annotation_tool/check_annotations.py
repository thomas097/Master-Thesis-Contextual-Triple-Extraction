from glob import glob
import random
import json

ARGUMENTS = ['subj', 'pred', 'obj', 'polar', 'cert']

if __name__ == '__main__':
    filenames = random.sample(glob('annotations/*.json'), 3)
    #filenames = ['JSON FILE']

    for fname in filenames:
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            tokens = data['tokens']
            annots = data['annotations']
            skipped = data['skipped']

            print('######' * 3 + ' %s ' % fname + '######' * 3)
            for turn in tokens:
                print(' '.join(turn))
            print()

            for triple in annots:
                if triple[0] or triple[1] or triple[2]: # has at least one argument
                    triple_args = []
                    for arg_name, arg in zip(ARGUMENTS, triple):
                        arg_string = ' '.join([tokens[i][j] for i,j in arg])
                        triple_args.append(arg_string)
                    print(triple_args)
            print()