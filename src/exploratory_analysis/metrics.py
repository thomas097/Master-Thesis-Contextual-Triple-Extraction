import spacy


## Pronominal expression types

def pronoun_count(doc):
    for token in doc:
        if token.pos_ == 'PRON':
            yield token.lower_


## Fragment / Ellipses / NSU types

AUX_LEMMAS = ['do', 'can', 'will', 'be', 'have', 'would', 'shall', 'should', 'may', 'might', 'must', 'could']
WH_WORDS = ['who', 'where', 'when', 'how', 'whose', 'what', 'why', 'which']


def is_VP_ellipsis(sent):
    for token in sent:
        if token.dep_ == 'conj':
            conjunct = list(token.subtree)
            has_aux = False
            has_verb = False
            for ctoken in conjunct:
                if ctoken.tag_ == 'MD' or ctoken.lemma in AUX_LEMMAS:
                    has_aux = True
                elif ctoken.pos_ == 'VERB':
                    has_verb = True

            if has_aux and not has_verb:
                return True
    return False


def is_sluice(sent):
    # Strip punctuation
    tokens = [t for t in sent if t.pos_ != 'PUNCT']

    # Just a Wh-word?
    if len(tokens) == 1 and tokens[0].lemma_ in WH_WORDS:
        return True

    # Indirect interrogative
    if tokens and tokens[-1].lemma_ in WH_WORDS:
        return True
    return False


def is_nominal_ellipsis(sent):
    tokens = list(sent)
    tokens += [tokens[-1]] # duplicate last just in case
    for i, token in enumerate(tokens[:-1]):
        if token.pos_ in ['NUM', 'DET'] and tokens[i + 1].pos_ not in ['PROPN', 'NOUN']:
            return True
    return False


def is_answer_ellipsis(sent):
    for token in sent:
        if token.dep_ == 'ROOT' and token.pos_ not in ['AUX', 'VERB']:
            return True
    return False


def fragment_count(doc):
    for sent in doc.sents:
        if is_VP_ellipsis(sent):
            yield 'VP-ellipses'
        elif is_sluice(sent):
            yield 'Sluice'
        elif is_nominal_ellipsis(sent):
            yield 'Nominal ellipsis'
        elif is_answer_ellipsis(sent):
            yield 'Answer ellipsis'
        else:
            yield 'Non-fragment'


def token_count(doc):
    for token in doc:
        yield token.lower_


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    print('VP-ellipsis? (True): ', is_VP_ellipsis(nlp('she will sell sea shells and he will too')))
    print('VP-ellipsis? (False):', is_VP_ellipsis(nlp('she will sell sea shells and he will do it too')))
    print('Sluice? (True): ', is_sluice(nlp('Did what?')))
    print('Sluice? (False):', is_sluice(nlp('What did she do?')))
    print('Nom-ellipsis? (True): ', is_nominal_ellipsis(nlp('I have a cat and Bob has three')))
    print('Nom-ellipsis? (False):', is_nominal_ellipsis(nlp('I have a cat and Bob has three cats')))
    print('Answer ellipsis? (True): ', is_answer_ellipsis(nlp('a lot of cats')))
    print('Answer ellipsis? (False):', is_answer_ellipsis(nlp('I have a lot of cats')))
