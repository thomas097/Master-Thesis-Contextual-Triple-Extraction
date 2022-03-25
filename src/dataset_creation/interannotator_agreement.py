from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import jaccard_distance


def two_coder_agreement(ann1, ann2):
    # Create matrix with, annotators, and triple labels
    task_data = []
    # Annotator 1
    for dialogue_id, triples in ann1.items():
        task_data.append(('coder1', dialogue_id, frozenset(triples)))
    # Annotator 2
    for dialogue_id, triples in ann2.items():
        task_data.append(('coder2', dialogue_id, frozenset(triples)))

    # Compute annotator agreement using Krippendorf
    task = AnnotationTask(distance=jaccard_distance)
    task.load_array(task_data)
    # No chance correction needed as random agreement is very unlikely (in the order of 1e-6)
    return task.avg_Ao()


if __name__ == '__main__':
    ann1 = {'d1': [('I', 'like', 'cats'), ('I', 'have', 'dogs')],
            'd2': [('I', 'like', 'cats'), ('I', 'have', 'dogs')],
            'd4': [('I', 'like', 'cats'), ('I', 'have', 'dogs')],
            'd3': [('I', 'like', 'cats'), ('I', 'have', 'dogs')]}
    ann2 = {'d1': [('I', 'like', 'cats'), ('I', 'have', 'dogs'), ('I', 'have', 'pandas')],
            'd2': [('I', 'like', 'cats'), ('I', 'have', 'dogs')],
            'd4': [('I', 'like', 'cats'), ('I', 'have', 'dogs')],
            'd3': [('I', 'like', 'cats'), ('I', 'have', 'cats')]}

    print(two_coder_agreement(ann1, ann2))
