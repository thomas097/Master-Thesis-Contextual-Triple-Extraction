from argument_extraction import ArgumentExtraction
from triple_scoring import TripleScoring
from utils import *

from itertools import product

if __name__ == '__main__':
    # Load models
    print('Loading models')
    arg_extract_model = ArgumentExtraction(base_model='albert-base-v2',
                                           path='models/argument_extraction_albert-v2_03_03_2022.pt')
    scoring_model = TripleScoring(base_model='albert-base-v2',
                                  path='models/scorer_albert-v2_03_03_2022.pt')
    print('\t- done!\n')

    # Test!
    input_ = 'i enjoy watching fish and like to make homework . <eos> what do you like ? <eos> animals , but i hate cats <eos>'
    print("Input: ", input_, '\n')

    # What arguments were found?
    y_subjs, y_preds, y_objs, subwords = arg_extract_model.predict(input_.split())
    subjs = bio_tags_to_tokens(subwords, y_subjs.T, one_hot=True)
    preds = bio_tags_to_tokens(subwords, y_preds.T, one_hot=True)
    objs = bio_tags_to_tokens(subwords, y_objs.T, one_hot=True)

    print('subjs: ', subjs)
    print('preds: ', preds)
    print('objs:  ', objs, '\n')

    # Score all possible triples
    triple_scores = []
    for subj, pred, obj in product(subjs, preds, objs):
        entailment, polarity, _ = scoring_model.predict(input_.split(), [subj, pred, obj])
        triple_scores.append((entailment, (subj, pred, obj, polarity)))

    # Rank triples from high (entailed) to low (not entailed)
    for score, triple in sorted(triple_scores, key=lambda x: -x[0]):
        print(score, triple)
