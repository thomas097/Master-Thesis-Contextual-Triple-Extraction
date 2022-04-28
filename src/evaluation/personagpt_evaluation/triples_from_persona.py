import spacy
from post_processing import PostProcessor
from tqdm import tqdm


def extract_triple(persona_line, nlp, post):
    """ Extract a triple from a persona line

    :param persona_line: string part of a persona
    :param nlp:          SpaCy instance
    :param post:         PostProcessor instance
    :return:             (subj, pred, obj, polarity) triple
    """
    doc = nlp(persona_line.strip().lower())

    # Skip triple if complex (not a single assertion)
    conjucts = [t for t in doc if t.dep_ == 'conj']
    if conjucts:
        return False

    # Polarity
    negations = any([t for t in doc if t.dep_ == 'neg'])
    polarity = 'negative' if negations else 'positive'

    # Subject
    subjs = [chunk for chunk in doc.noun_chunks if 'subj' in chunk.root.dep_]
    if len(subjs) == 1:
        subj = list(subjs[0])
    else:
        return False  # Multiple or no subjects

    # Object
    objs = [token.subtree for token in doc if 'obj' in token.dep_]
    if len(objs) == 1:
        obj_ = list(objs[0])
    else:
        return False  # Multiple or no objects

    # Predicate
    pred = [t for t in doc if t not in subj + obj_ and t.pos_ != 'PUNCT']
    if not pred or len(pred) > 3:
        return False  # No predicate somehow

    # Format back to string
    subj = ' '.join([t.lower_ for t in subj if t.dep_ != 'neg'])
    pred = ' '.join([t.lower_ for t in pred if t.dep_ != 'neg'])
    obj_ = ' '.join([t.lower_ for t in obj_ if t.dep_ != 'neg'])
    return post.format((subj, pred, obj_)) + (polarity,)


if __name__ == '__main__':
    # Personas are simple enough to parse with SpaCy
    nlp = spacy.load('en_core_web_sm')
    post = PostProcessor()

    # Lets sample 10.000
    N = 100000

    # Define output file
    outfile = open('persona_triples.txt', 'w', encoding='utf-8')

    with open('persona_lines.txt', 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()[:N]):
            # Try to extract triple from persona (ignore complex ones!)
            triple = extract_triple(line, nlp=nlp, post=post)
            if triple:
                outfile.write("\"{}\", {}\n".format(line.strip(), triple))

    outfile.close()
