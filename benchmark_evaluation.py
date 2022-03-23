import sys
sys.path.insert(0, '../model_dependency')
sys.path.insert(0, '../model_transformer')

from run_transformer_pipeline import AlbertTripleExtractor
from utils import load_annotations, triple_to_bio_tags


def triples_from_annotation(ann):
    triple_lst = ann['triples'] if 'triples' in ann else ann['annotations']  # TODO: fix
    tokens = ann['tokens']

    triples = []
    for triple in triple_lst:
        subj = ' '.join([tokens[i][j] for i, j in triple[0]])
        pred = ' '.join([tokens[i][j] for i, j in triple[1]])
        obj_ = ' '.join([tokens[i][j] for i, j in triple[2]])
        polar = 'positive' if not triple[3] else 'negative'

        # Skip blank triples
        if subj or pred or obj_:
            triples.append((subj, pred, obj_, polar))
    return triples


def best_match(preds, labels):
    return list(zip(preds, labels))  # TODO: match with e.g. one word margin


def evaluate(annotation_file, model, decision_thres=0.5):
    # Extract triples from annotations
    data = []
    for ann in load_annotations(annotation_file):
        # Ground truth
        y_true = triples_from_annotation(ann)

        # Predictions
        input_ = ' '.join([t for ts in ann['tokens'] for t in ts + ['<eos>']])
        y_pred = [triple for ent, triple in model.extract_triples(input_) if ent > decision_thres]

        # Match correctly predicted
        data += best_match(y_true, y_pred)
        break

    print(data)




if __name__ == '__main__':
    model = AlbertTripleExtractor('../model_transformer/models/argument_extraction_albert-v2_17_03_2022',
                                  '../model_transformer/models/scorer_albert-v2_17_03_2022')
    evaluate('../annotation_tool/dev_annotations', model)  # TODO: test set
