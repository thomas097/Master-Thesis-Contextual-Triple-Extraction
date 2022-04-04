import os
import json
import glob

import spacy


def load_annotations(path):
    annotations = dict()
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations[fname] = data
    return annotations


def triples(annotation):
    tokens = annotation['tokens']
    for subj, pred, obj, _, _ in annotation['annotations']:
        subj = ' '.join([tokens[i][j] for i, j in subj])
        pred = ' '.join([tokens[i][j] for i, j in pred])
        obj = ' '.join([tokens[i][j] for i, j in obj])
        yield subj, pred, obj


class AutoNormalizer:
    def __init__(self):
        self._nlp = spacy.load('en_core_web_sm')

        self._simple_cases = {'like': ['like', 'enjoy', 'like to', 'enjoy to'],
                              'love': ['love', 'in love with'],
                              'do_activity': ['do for fun', 'like to do'],
                              'work_for': ['employed by', 'work for'],
                              'dislike': ['hate', 'dislike']}

        # Always ask annotator for these!
        self._ambiguous_cases = {'be', 'has', 'is', 'have', 'go', 'going'}

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

        if pos_seq == 'VERB ADP' or pos_seq == 'VERB ADV':  # 'share with' -> 'share_with'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB PART':  # 'going to' -> 'go_to'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB ADP DET NOUN':  # 'meet at the airport' -> 'meet_at_the_airport'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'VERB':  # 'driven' -> 'drive'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX VERB PART':  # 'am going to' -> 'go_to'
            pred = '_'.join(lemmas[1:])
            return pred, True

        if pos_seq == 'INTJ PART VERB' or pos_seq == 'VERB PART VERB':  # 'like to hit' -> 'likes_to_hit'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'INTJ ADP NOUN':  # 'like with dinner' -> 'likes_with_dinner'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX ADP':  # 'is about' -> 'be_about'
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq.count(' ') > 4:  # 'likes to read books about' -> 'likes_to_read_books_about' (very specific)
            pred = '_'.join(lemmas)
            return pred, True

        if pos_seq == 'AUX VERB ADP': # 'was diagnosed with' -> 'diagnosed_with'
            pred = '_'.join(lemmas[1:])
            return pred, True

        if pos_seq == 'ADJ ADP':
            pred = 'is_' + ('_'.join(lemmas[1:]))
            return pred, True

        if pos_seq == 'AUX VERB':
            pred = lemmas[1]
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
        annot['norm_predicates'] = []
        annot['annotations'] = [lst for lst in annot['annotations'] if any(lst)]
        for triple in triples(annot):
            print('\n', triple)

            # Try to normalize using simple rules
            norm_pred, changed = normalizer.normalize(triple[1])

            # If not auto-normalized, ask annotator
            if not changed:
                norm_pred = input('> ')
            else:
                print(norm_pred)

            annot['norm_predicates'].append(norm_pred)

        # Write back to file
        with open(new_fname, 'w', encoding='utf-8') as file:
            json.dump(annot, file)





if __name__ == '__main__':
    INPUT_DIR = 'trainval_annotations_thomas'
    OUTPUT_DIR = 'normalized_annotations'
    main(INPUT_DIR, OUTPUT_DIR)


