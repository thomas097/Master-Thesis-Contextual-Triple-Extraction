import os
import json
import glob
import spacy
from nltk.corpus import wordnet as wn
from tqdm import tqdm

PRED_RULES = []


def load_annotations(path):
    annotations = dict()
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations[fname] = data
    return annotations


def triples(annotation):
    tokens = annotation['tokens']
    for subj, pred, obj, neg, cert in annotation['annotations']:
        subj = ' '.join([tokens[i][j] for i, j in subj])
        pred = ' '.join([tokens[i][j] for i, j in pred])
        obj = ' '.join([tokens[i][j] for i, j in obj])
        neg = 'negative' if neg else 'positive'
        cert = 'uncertain' if cert else 'certain'
        yield subj, pred, obj, neg, cert


def arg_to_synset(arg, nlp):
    # Easy cases (pronouns)
    if arg.lower() in ['i', 'you', 'he', 'she', 'him', 'her', 'me']:
        return wn.synset('person.n.01').name()
    if arg.lower() in ['they', 'them', 'we']:
        return wn.synset('group.n.01').name()
    if not arg or arg.lower() in ['this', 'that', 'those', 'them']:
        return 'UNK'

    doc = nlp(arg)
    tokens = [t for t in doc if t.dep_ in ['ROOT'] and t.pos_ in ['NOUN', 'VERB', 'PROPN', 'ADJ']]
    for token in tokens[::-1]:
        synset_matches = wn.synsets(token.lemma_)
        if synset_matches:
            return synset_matches[0].name()

    if arg[0].isnumeric():
        return wn.synset('quantity.n.01').name()

    for ent in doc.ents:
        if (ent.end_char - ent.start_char) / len(arg) > 0.7:
            if ent.label_ == 'MONEY':
                return wn.synset('money.n.01').name()
            if ent.label_ == 'PERSON':
                return wn.synset('person.n.01').name()
            if ent.label_ == 'ORG':
                return wn.synset('organization.n.01').name()
            if ent.label_ == 'GEO':
                return wn.synset('topographic_point.n.01').name()
            if ent.label_ == 'EVE':
                return wn.synset('event.n.01').name()
    return 'UNK'


def pred_hypernyms(synset_name):
    hierarchy = [wn.synset(synset_name, pos=wn.VERB)]
    while wn.hypernyms(hierarchy[-1]):
        hierarchy.append(wn.hypernyms(hierarchy[-1])[0])
    return hierarchy


def get_synset_name(synset):
    holonyms = synset.member_holonyms()
    if not holonyms:
        return synset.name()
    return min(holonyms, key=lambda x: int(x[-2:])).name()


def pred_to_synset(pred, nlp, subj_synset, obj_synset):
    # Determine main verb
    doc = [t for t in nlp('I ' + pred + ' it')][1:-1]
    tokens = [t.lemma_ for t in doc if t.dep_ in ['ROOT'] and t.pos_ in ['VERB', 'AUX', 'INTJ']]
    for lemma in tokens:
        # Check if there are any rules
        for (subj_type, pred_lemma, obj_type, pred_synset) in PRED_RULES:
            if subj_type in pred_hypernyms(subj_synset) and obj_type in pred_hypernyms(obj_synset) and pred == pred_lemma:
                return get_synset_name(pred_synset)

        # Base case
        synsets = wn.synsets(lemma, pos=wn.VERB)
        if synsets:
            return get_synset_name(synsets[0])
    return 'UNK'


if __name__ == '__main__':
    INPUT_DIR = 'trainval_annotations_thomas'
    OUTPUT_DIR = 'trainval_annotations_thomas_normalized'

    nlp = spacy.load('en_core_web_sm')

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    for fname, ann in tqdm(load_annotations(INPUT_DIR).items()):

        ann['synsets'] = []
        for subj, pred, obj, polar, cert in triples(ann):
            subj_type = arg_to_synset(subj, nlp)
            obj_type = arg_to_synset(obj, nlp)
            pred_type = pred_to_synset(pred, nlp, subj_type, obj_type)

            ann['synsets'].append([subj_type, pred_type, obj_type])

        # Save to file
        new_fname = fname.replace(INPUT_DIR, OUTPUT_DIR)
        with open(new_fname, 'w', encoding='utf-8') as file:
            json.dump(ann, file)







