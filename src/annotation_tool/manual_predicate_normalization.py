import os
import json
import glob
import spacy
from lemminflect import getInflection


def load_annotations(path):
    annotations = dict()
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations[fname] = data
    return annotations


def to_user_id(token, i):
    token = token.lower().strip()
    if i % 2 == 0: # first and last turn -> speaker1
        if token == 'i' or token == 'me' or token == 'myself':
            token = '[speaker1]'
        elif token == 'my' or token == 'mine':
            token = '[speaker1] \'s'
        elif token == 'you' or token == 'yourself':
            token = '[speaker2]'
        elif token == 'your' or token == 'yours':
            token = '[speaker2] \'s'
    else:
        if token == 'i' or token == 'me' or token == 'myself':
            token = '[speaker2]'
        elif token == 'my' or token == 'mine':
            token = '[speaker2] \'s'
        elif token == 'you' or token == 'yourself':
            token = '[speaker1]'
        elif token == 'your' or token == 'yours':
            token = '[speaker1] \'s'
    return token


def triples(annotation):
    tokens = annotation['tokens']
    for subj, pred, obj, neg, cert in annotation['annotations']:
        subj = ' '.join([to_user_id(tokens[i][j], i) for i, j in subj])
        pred = ' '.join([to_user_id(tokens[i][j], i) for i, j in pred])
        obj = ' '.join([to_user_id(tokens[i][j], i) for i, j in obj])
        neg = 'negative' if neg else 'positive'
        cert = 'uncertain' if cert else 'certain'
        yield subj, pred, obj, neg, cert


class AutoNormalizer:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')

        self._simple_cases = {'likes': ['like', 'enjoy', 'like to', 'enjoy to', 'prefers',
                                        'prefer'],
                              'loves': ['love', 'in love with'],
                              'does_activity': ['do for fun', 'like to do'],
                              'works_for': ['employed by', 'work for'],
                              'dislikes': ['hate', 'dislike']}

        # Always ask annotator for these!
        self._ambiguous_cases = {'be', 'has', 'is', 'have', 'go', 'going'}

        self._special_auxs = ['should', 'can', 'could', 'must', 'will', 'would']

    def normalize(self, pred):
        pos_seq = ' '.join([t.pos_ for t in self._nlp(pred)])
        lemmas = [t.lemma_ for t in self._nlp(pred)]

        # If too simple, ask annotator
        if ' '.join(lemmas) in self._ambiguous_cases:
            return pred, False

        # Check if we can apply a simple rule
        for norm_pred, simple_preds in self._simple_cases.items():
            if ' '.join(lemmas) in simple_preds:
                return norm_pred, True  # True to indicate a normalization

        if pos_seq == 'VERB ADP' or pos_seq == 'VERB ADV':  # 'share with' -> 'shares_with'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB PART':  # 'going to' -> 'goes_to'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB ADP DET NOUN':  # 'meet at the airport' -> 'meets_at_the_airport'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB':  # 'driven' -> 'drives'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX VERB PART':  # 'am going to' -> 'goes_to'
            lemmas[1] = getInflection(lemmas[1], tag='VBZ')[0]
            pred = '_'.join(lemmas[1:])
            return pred, True

        if pos_seq == 'INTJ PART VERB' or pos_seq == 'VERB PART VERB':  # 'like to hit' -> 'likes_to_hit'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'INTJ ADP NOUN':  # 'like with dinner' -> 'likes_with_dinner'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX ADP':  # 'is about' -> 'is_about'
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq.count(' ') > 4 and pos_seq.startswith('VERB'):  # 'likes to read books about' -> 'likes_to_read_books_about' (very specific)
            lemmas[0] = getInflection(lemmas[0], tag='VBZ')[0]
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX VERB ADP': # 'was diagnosed with' -> 'diagnosed_with'
            pred = '_'.join(lemmas[1:])
            return pred, True

        if pos_seq == 'ADJ ADP':
            pred = 'is_' + ('_'.join(lemmas[1:]))
            return pred, True

        if pos_seq == 'AUX VERB' and lemmas[0] not in self._special_auxs:
            pred = getInflection(lemmas[1], tag='VBZ')[0]
            return pred, True

        print('POS:', pos_seq)
        return pred, False


def main(input_dir, output_dir):
    # Create new directory to store results into
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create normalizer to do all the simple cases
    normalizer = AutoNormalizer()

    # Load annotations to mark
    annotations = load_annotations(input_dir)
    for fname, annot in annotations.items():

        # If already done, continue with a next file
        new_fname = fname.replace(input_dir, output_dir)
        if os.path.isfile(new_fname):
            continue

        # Augment annotations var with additional normalized predicates
        annot['norm_triples'] = []
        annot['annotations'] = [lst for lst in annot['annotations'] if any(lst)]
        for subj, pred, obj, pol, cert in triples(annot):
            print('\n', (subj, pred, obj))

            # Try to normalize using simple rules
            norm_pred, changed = normalizer.normalize(pred)

            # If not auto-normalized, ask annotator

            if not changed:
                norm_pred = input('> ')
            else:
                print(norm_pred)

            annot['norm_triples'].append((subj, norm_pred, obj, pol, cert))

        # Write back to file
        with open(new_fname, 'w', encoding='utf-8') as file:
            json.dump(annot, file)





if __name__ == '__main__':
    INPUT_DIR = 'trainval_annotations_thomas'
    OUTPUT_DIR = 'normalized_annotations'
    main(INPUT_DIR, OUTPUT_DIR)


